use anyhow::{Context, Result, bail};
use rusqlite::{Connection, TransactionBehavior};

use crate::Store;

const MIGRATIONS: &[&str] = &[
    r#"
    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE procedures (
        rowid INTEGER PRIMARY KEY,
        id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        family TEXT NOT NULL,
        repo TEXT NOT NULL,
        state TEXT NOT NULL,
        title TEXT NOT NULL,
        signature TEXT NOT NULL,
        document TEXT NOT NULL,
        embedding BLOB,
        updated_at TEXT NOT NULL,
        UNIQUE (id, revision)
    );
    CREATE INDEX procedures_family ON procedures (family);
    CREATE INDEX procedures_repo ON procedures (repo, state);

    CREATE VIRTUAL TABLE procedures_fts USING fts5 (
        title, body,
        tokenize = "unicode61 tokenchars '_-./'"
    );

    CREATE TABLE entities (
        key TEXT NOT NULL,
        kind TEXT NOT NULL,
        procedure INTEGER NOT NULL REFERENCES procedures (rowid) ON DELETE CASCADE
    );
    CREATE INDEX entities_key ON entities (key);
    CREATE INDEX entities_procedure ON entities (procedure);

    CREATE TABLE extractions (
        session TEXT NOT NULL,
        first_seq INTEGER NOT NULL,
        summary TEXT NOT NULL,
        procedure INTEGER REFERENCES procedures (rowid) ON DELETE SET NULL,
        rejection TEXT,
        at TEXT NOT NULL,
        PRIMARY KEY (session, first_seq)
    );

    CREATE TABLE sessions (
        session TEXT PRIMARY KEY,
        harness TEXT NOT NULL,
        transcript TEXT NOT NULL,
        repo TEXT,
        extracted_through INTEGER NOT NULL,
        ended INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE repos (
        root TEXT PRIMARY KEY,
        repo TEXT NOT NULL
    );

    CREATE TABLE injections (
        rowid INTEGER PRIMARY KEY,
        session TEXT NOT NULL,
        procedure TEXT NOT NULL,
        revision INTEGER NOT NULL,
        holdout INTEGER NOT NULL,
        at TEXT NOT NULL
    );
    CREATE INDEX injections_session ON injections (session, procedure);
"#,
    r"
    CREATE VIRTUAL TABLE procedures_vocab USING fts5vocab (procedures_fts, 'row');
    INSERT INTO procedures_fts (procedures_fts, rank) VALUES ('rank', 'bm25(2.0, 1.0)');
",
    r"
    CREATE TABLE sessions_rebuilt (
        session TEXT PRIMARY KEY,
        harness TEXT NOT NULL,
        transcript TEXT NOT NULL,
        repo TEXT,
        extracted_through INTEGER,
        ended INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );
    INSERT INTO sessions_rebuilt
        SELECT session, harness, transcript, repo, extracted_through, ended, updated_at FROM sessions;
    DROP TABLE sessions;
    ALTER TABLE sessions_rebuilt RENAME TO sessions;
",
    r"
    ALTER TABLE procedures ADD COLUMN used_at TEXT;
    UPDATE procedures SET used_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now');

    CREATE TABLE embeddings (
        procedure INTEGER NOT NULL REFERENCES procedures (rowid) ON DELETE CASCADE,
        embedding BLOB NOT NULL
    );
    CREATE INDEX embeddings_procedure ON embeddings (procedure);
    INSERT INTO embeddings (procedure, embedding)
        SELECT rowid, embedding FROM procedures WHERE embedding IS NOT NULL;
    ALTER TABLE procedures DROP COLUMN embedding;

    ALTER TABLE injections ADD COLUMN cue TEXT NOT NULL DEFAULT 'prompt';
    ALTER TABLE injections ADD COLUMN task INTEGER;
    ALTER TABLE injections ADD COLUMN outcome TEXT;
    CREATE INDEX injections_outcomes ON injections (procedure, holdout, outcome);

    ALTER TABLE extractions ADD COLUMN outcome TEXT;
    ALTER TABLE extractions ADD COLUMN tool_calls INTEGER;
    ALTER TABLE extractions ADD COLUMN started_at TEXT;
    ALTER TABLE extractions ADD COLUMN ended_at TEXT;
",
];

impl Store {
    pub(crate) fn is_current(&self) -> Result<bool> {
        Ok(applied_migrations(&self.conn)? == MIGRATIONS.len())
    }

    pub(crate) fn migrate(&mut self) -> Result<()> {
        if self.is_current()? {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("lock the database for migration")?;
        let applied = applied_migrations(&tx)?;
        for (index, migration) in MIGRATIONS.iter().enumerate().skip(applied) {
            tx.execute_batch(migration)
                .with_context(|| format!("migrate the schema to version {}", index + 1))?;
        }
        let latest = i64::try_from(MIGRATIONS.len()).context("count schema versions")?;
        tx.pragma_update(None, "user_version", latest)
            .context("record the schema version")?;
        tx.commit().context("commit the migration")
    }
}

fn applied_migrations(conn: &Connection) -> Result<usize> {
    let version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .context("read the schema version")?;
    let applied = usize::try_from(version).context("read a valid schema version")?;
    if applied > MIGRATIONS.len() {
        bail!(
            "this database was created by a newer version of trodden (schema version {applied}, \
             this build knows up to {}); upgrade trodden to open it",
            MIGRATIONS.len()
        );
    }
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        sync::{Arc, Barrier},
        thread,
    };

    use crate::Patience;

    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("trodden-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch directory is writable");
        dir
    }

    fn version(store: &Store) -> i64 {
        store
            .conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .expect("schema version reads")
    }

    #[test]
    fn concurrent_opens_of_a_fresh_database_all_succeed() {
        let dir = scratch("concurrent-opens");
        for round in 0..40 {
            let path = dir.join(format!("{round}.db"));
            let barrier = Arc::new(Barrier::new(8));
            let openers: Vec<_> = (0..8)
                .map(|_| {
                    let path = path.clone();
                    let barrier = Arc::clone(&barrier);
                    thread::spawn(move || {
                        barrier.wait();
                        Store::open(&path, Patience::Batch).map(|store| version(&store))
                    })
                })
                .collect();
            for opener in openers {
                let opened = opener.join().expect("opener does not panic");
                assert_eq!(
                    opened.map_err(|error| format!("{error:#}")),
                    Ok(i64::try_from(MIGRATIONS.len()).expect("few migrations")),
                    "round {round}",
                );
            }
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_database_from_a_newer_version_is_refused() {
        let dir = scratch("newer-version");
        let path = dir.join("trodden.db");
        let store = Store::open(&path, Patience::Batch).expect("store opens");
        store
            .conn
            .pragma_update(None, "user_version", 99)
            .expect("schema version writes");
        drop(store);

        let refusal = Store::open(&path, Patience::Batch).expect_err("newer schema is refused");

        assert!(
            format!("{refusal:#}").contains("created by a newer version of trodden"),
            "{refusal:#}",
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
