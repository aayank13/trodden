mod ingest;
mod learning;
mod procedures;
mod schema;
mod terms;

use std::{
    path::Path,
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use jiff::Timestamp;
use rusqlite::{Connection, ErrorCode, OpenFlags, OptionalExtension, params};

pub use ingest::{ExtractionRecord, Progress};
pub use learning::{Cue, FamilyEvidence, Injection, InjectionRecord, OutcomeSummary, Usage};
pub use procedures::{EntityHit, Forget, LexicalHit, ProcedureRow, Upsert};
pub use terms::Terms;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Patience {
    Batch,
    Interactive,
}

impl Patience {
    fn timeout(self) -> Duration {
        match self {
            Self::Batch => Duration::from_secs(5),
            Self::Interactive => Duration::from_millis(50),
        }
    }
}

#[derive(Debug)]
pub struct Store {
    conn: Connection,
}

impl Store {
    pub fn open(path: &Path, patience: Patience) -> Result<Self> {
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE
            | OpenFlags::SQLITE_OPEN_CREATE
            | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(path, flags)
            .with_context(|| format!("open {}", path.display()))?;
        conn.busy_timeout(patience.timeout())
            .context("set the busy timeout")?;
        if patience == Patience::Batch {
            Self::enable_wal(&conn, patience.timeout())?;
            conn.execute_batch("PRAGMA foreign_keys = ON;")
                .context("configure the database")?;
        }
        conn.execute_batch("PRAGMA synchronous = NORMAL;")
            .context("configure durability")?;
        let mut store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn open_read_only(path: &Path, patience: Patience) -> Result<Self> {
        let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        let conn = Connection::open_with_flags(path, flags)
            .with_context(|| format!("open {} for reading", path.display()))?;
        conn.busy_timeout(patience.timeout())
            .context("set the busy timeout")?;
        let store = Self { conn };
        if !store.is_current()? {
            bail!(
                "the database at {} needs an upgrade before it can be read; \
                 run any trodden command, such as `trodden status`, to upgrade it",
                path.display()
            );
        }
        Ok(store)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().context("open an in-memory database")?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .context("configure the database")?;
        let mut store = Self { conn };
        store.migrate()?;
        Ok(store)
    }

    pub fn is_busy(error: &anyhow::Error) -> bool {
        error.chain().any(|cause| {
            matches!(
                cause
                    .downcast_ref::<rusqlite::Error>()
                    .and_then(rusqlite::Error::sqlite_error_code),
                Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
            )
        })
    }

    pub fn paused(&self) -> Result<bool> {
        Ok(self.setting("paused")?.as_deref() == Some("1"))
    }

    pub fn set_paused(&self, paused: bool) -> Result<()> {
        self.set_setting("paused", if paused { "1" } else { "0" })
    }

    pub fn holdout_rate(&self) -> Result<Option<f64>> {
        self.setting("holdout")?
            .map(|rate| rate.parse().context("parse the holdout rate"))
            .transpose()
    }

    pub fn set_holdout_rate(&self, rate: Option<f64>) -> Result<()> {
        match rate {
            Some(rate) => self.set_setting("holdout", &rate.to_string()),
            None => self.clear_setting("holdout"),
        }
    }

    pub fn verify_reminder(&self) -> Result<bool> {
        Ok(self.setting("verify_reminder")?.as_deref() == Some("1"))
    }

    pub fn set_verify_reminder(&self, on: bool) -> Result<()> {
        self.set_setting("verify_reminder", if on { "1" } else { "0" })
    }

    pub fn forgotten_before(&self) -> Result<Option<Timestamp>> {
        self.setting("forgotten_before")?
            .map(|at| {
                at.parse()
                    .context("parse the time everything was forgotten")
            })
            .transpose()
    }

    pub fn embedding_scheme(&self) -> Result<Option<String>> {
        self.setting("embedding_scheme")
    }

    pub fn set_embedding_scheme(&self, scheme: &str) -> Result<()> {
        self.set_setting("embedding_scheme", scheme)
    }

    fn clear_setting(&self, key: &str) -> Result<()> {
        self.conn
            .execute("DELETE FROM settings WHERE key = ?1", [key])
            .with_context(|| format!("clear setting {key}"))?;
        Ok(())
    }

    fn setting(&self, key: &str) -> Result<Option<String>> {
        self.conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()
            .with_context(|| format!("read setting {key}"))
    }

    fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO settings (key, value) VALUES (?1, ?2)
                 ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                params![key, value],
            )
            .with_context(|| format!("write setting {key}"))?;
        Ok(())
    }

    pub fn repo_for_root(&self, root: &str) -> Result<Option<String>> {
        self.conn
            .query_row("SELECT repo FROM repos WHERE root = ?1", [root], |row| {
                row.get(0)
            })
            .optional()
            .context("look up a repository id")
    }

    pub fn remember_repo(&self, root: &str, repo: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO repos (root, repo) VALUES (?1, ?2)
                 ON CONFLICT (root) DO UPDATE SET repo = excluded.repo",
                params![root, repo],
            )
            .context("cache a repository id")?;
        Ok(())
    }

    pub fn stats(&self) -> Result<Stats> {
        self.conn
            .query_row(
                "SELECT
                    (SELECT COUNT(DISTINCT id) FROM procedures WHERE state != 'retired'),
                    (SELECT COUNT(*) FROM procedures),
                    (SELECT COUNT(*) FROM sessions),
                    (SELECT COUNT(*) FROM extractions WHERE rejection IS NOT NULL),
                    (SELECT COUNT(*) FROM injections WHERE NOT holdout)",
                [],
                |row| {
                    let count = |index| {
                        row.get::<_, i64>(index)
                            .map(|n| u64::try_from(n).unwrap_or(0))
                    };
                    Ok(Stats {
                        procedures: count(0)?,
                        revisions: count(1)?,
                        sessions: count(2)?,
                        rejections: count(3)?,
                        injections: count(4)?,
                    })
                },
            )
            .context("count store contents")
    }

    fn enable_wal(conn: &Connection, timeout: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        loop {
            match conn
                .pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get::<_, String>(0))
            {
                Ok(_) => return Ok(()),
                Err(error)
                    if error.sqlite_error_code() == Some(ErrorCode::DatabaseBusy)
                        && Instant::now() < deadline =>
                {
                    thread::sleep(Duration::from_millis(5));
                }
                Err(error) => return Err(error).context("switch the database to WAL mode"),
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub procedures: u64,
    pub revisions: u64,
    pub sessions: u64,
    pub rejections: u64,
    pub injections: u64,
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use super::*;

    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("trodden-read-only-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch directory is writable");
            Self { dir }
        }

        fn database(&self) -> PathBuf {
            self.dir.join("trodden.db")
        }

        fn fingerprint(&self) -> (Vec<u8>, i64) {
            let path = self.database();
            let contents = fs::read(&path).expect("database reads");
            let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .expect("database opens");
            let version = conn
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .expect("schema version reads");
            (contents, version)
        }

        fn set_version(&self, version: i64) {
            let store = Store::open(&self.database(), Patience::Batch).expect("store opens");
            store
                .conn
                .pragma_update(None, "user_version", version)
                .expect("schema version writes");
            store
                .conn
                .pragma_update_and_check(None, "wal_checkpoint", "TRUNCATE", |_| Ok(()))
                .expect("checkpoint runs");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn read_only_stores_read_without_writing() {
        let scratch = Scratch::new("reads");
        let store = Store::open(&scratch.database(), Patience::Batch).expect("store opens");
        store
            .remember_repo("/home/dev/shop", "shop")
            .expect("repo caches");
        store.set_paused(true).expect("setting writes");
        drop(store);
        let before = scratch.fingerprint();

        let reader =
            Store::open_read_only(&scratch.database(), Patience::Batch).expect("store opens");
        assert!(reader.paused().expect("setting reads"));
        assert_eq!(
            reader.repo_for_root("/home/dev/shop").expect("repo reads"),
            Some("shop".to_owned())
        );
        assert!(reader.remember_repo("/home/dev/blog", "blog").is_err());
        drop(reader);

        assert_eq!(scratch.fingerprint(), before);
    }

    #[test]
    fn read_only_stores_read_while_a_writer_holds_the_lock() {
        let scratch = Scratch::new("locked");
        let writer = Store::open(&scratch.database(), Patience::Batch).expect("store opens");
        writer.set_paused(true).expect("setting writes");
        writer
            .conn
            .execute_batch("BEGIN IMMEDIATE; INSERT INTO repos (root, repo) VALUES ('a', 'b');")
            .expect("writer locks the database");

        let reader = Store::open_read_only(&scratch.database(), Patience::Interactive)
            .expect("store opens while locked");
        assert!(reader.paused().expect("setting reads"));
        assert_eq!(reader.repo_for_root("a").expect("repo reads"), None);
        writer
            .conn
            .execute_batch("COMMIT;")
            .expect("writer commits");
    }

    #[test]
    fn a_write_blocked_by_another_writer_is_busy() {
        let scratch = Scratch::new("busy");
        let writer = Store::open(&scratch.database(), Patience::Batch).expect("store opens");
        writer
            .conn
            .execute_batch("BEGIN IMMEDIATE; INSERT INTO repos (root, repo) VALUES ('a', 'b');")
            .expect("writer locks the database");
        let blocked = Store::open(&scratch.database(), Patience::Interactive).expect("store opens");

        let busy = blocked
            .set_paused(true)
            .expect_err("the writer holds the lock");
        writer
            .conn
            .execute_batch("COMMIT;")
            .expect("writer commits");

        assert!(Store::is_busy(&busy), "{busy:#}");
        assert!(Store::is_busy(&busy.context("pause capture")));
        blocked
            .set_paused(true)
            .expect("setting writes once the lock is free");
    }

    #[test]
    fn other_store_errors_are_not_busy() {
        let scratch = Scratch::new("not-busy");
        drop(Store::open(&scratch.database(), Patience::Batch).expect("store opens"));
        let reader =
            Store::open_read_only(&scratch.database(), Patience::Batch).expect("store opens");

        let refusal = reader
            .set_paused(true)
            .expect_err("a read-only store cannot write");

        assert!(!Store::is_busy(&refusal), "{refusal:#}");
        assert!(!Store::is_busy(&anyhow::anyhow!("database is locked")));
    }

    #[test]
    fn read_only_stores_ask_for_an_upgrade_instead_of_migrating() {
        let scratch = Scratch::new("outdated");
        fs::write(scratch.database(), b"").expect("database file is writable");
        let empty = scratch.fingerprint();
        let refusal = Store::open_read_only(&scratch.database(), Patience::Batch)
            .expect_err("unmigrated database is refused");
        assert!(
            format!("{refusal:#}").contains("run any trodden command"),
            "{refusal:#}"
        );
        assert_eq!(scratch.fingerprint(), empty);

        scratch.set_version(1);
        let behind = scratch.fingerprint();
        let refusal = Store::open_read_only(&scratch.database(), Patience::Batch)
            .expect_err("outdated schema is refused");
        assert!(
            format!("{refusal:#}").contains("run any trodden command"),
            "{refusal:#}"
        );
        assert_eq!(scratch.fingerprint(), behind);
    }

    #[test]
    fn stats_count_only_extractions_with_a_rejection_reason() {
        let store = Store::open_in_memory().expect("store opens");
        for (first_seq, rejection) in [(0, None), (8, Some("no files were changed"))] {
            store
                .record_extraction(&ExtractionRecord {
                    session: "s_4b1d".to_owned(),
                    first_seq,
                    summary: "Page 2 repeats the last product".to_owned(),
                    procedure: None,
                    rejection: rejection.map(str::to_owned),
                    outcome: Some("succeeded".to_owned()),
                    tool_calls: Some(6),
                    span: None,
                    at: "2026-09-21T14:02:44Z".to_owned(),
                })
                .expect("extraction recorded");
        }

        let listed = store.rejections(10).expect("rejections read").len();

        assert_eq!(store.stats().expect("stats read").rejections, 1);
        assert_eq!(listed, 1);
    }

    #[test]
    fn read_only_stores_refuse_newer_and_missing_databases() {
        let scratch = Scratch::new("refused");
        let missing = Store::open_read_only(&scratch.database(), Patience::Batch);
        assert!(missing.is_err());
        assert!(!scratch.database().exists());

        scratch.set_version(99);
        let refusal = Store::open_read_only(&scratch.database(), Patience::Batch)
            .expect_err("newer schema is refused");
        assert!(
            format!("{refusal:#}").contains("created by a newer version of trodden"),
            "{refusal:#}"
        );
    }
}
