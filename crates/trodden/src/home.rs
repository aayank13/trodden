use std::{
    env,
    fmt::Display,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use jiff::Timestamp;
use trodden_embed::Embedder;
use trodden_recall::{Recall, VectorIndex};
use trodden_store::{Patience, Store};

#[derive(Debug, Clone)]
pub struct Home {
    dir: PathBuf,
}

impl Home {
    pub const ENV: &str = "TRODDEN_HOME";
    const ERROR_LOG_LIMIT: u64 = 64 * 1024;

    pub fn locate() -> Result<Self> {
        if let Some(dir) = env::var_os(Self::ENV).filter(|dir| !dir.is_empty()) {
            return Ok(Self {
                dir: PathBuf::from(dir),
            });
        }
        let base = if cfg!(windows) {
            env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            env::home_dir().map(|home| home.join("Library/Application Support"))
        } else {
            env::var_os("XDG_DATA_HOME")
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from)
                .or_else(|| env::home_dir().map(|home| home.join(".local/share")))
        };
        let base = base.context("find a data directory; set TRODDEN_HOME")?;
        Ok(Self {
            dir: base.join("trodden"),
        })
    }

    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn is_initialized(&self) -> bool {
        self.database().is_file()
    }

    pub fn initialize(&self) -> Result<Store> {
        fs::create_dir_all(&self.dir).with_context(|| format!("create {}", self.dir.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&self.dir, fs::Permissions::from_mode(0o700))
                .with_context(|| format!("restrict access to {}", self.dir.display()))?;
        }
        self.open_store(Patience::Batch)
    }

    pub fn database(&self) -> PathBuf {
        self.dir.join("trodden.db")
    }

    pub fn index(&self) -> PathBuf {
        self.dir.join("recall.index")
    }

    pub fn embeddings(&self) -> PathBuf {
        self.dir.join("embeddings.pack")
    }

    pub fn ingest_lock(&self) -> PathBuf {
        self.dir.join("ingest.lock")
    }

    pub fn pending_ingests(&self) -> PathBuf {
        self.dir.join("ingest.pending")
    }

    pub fn journals(&self) -> PathBuf {
        self.dir.join("journals")
    }

    pub fn journal(&self, harness: &str, session: &str) -> PathBuf {
        let plain = !session.is_empty()
            && session.len() <= 128
            && !session.starts_with('.')
            && session
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c));
        let name = if plain {
            session.to_owned()
        } else {
            let hash = session
                .bytes()
                .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                    (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
                });
            format!("session-{hash:016x}")
        };
        self.journals().join(harness).join(format!("{name}.jsonl"))
    }

    pub fn error_log(&self) -> PathBuf {
        self.dir.join("hook.log")
    }

    pub fn log_error(&self, error: impl Display) {
        if !self.is_initialized() {
            return;
        }
        let path = self.error_log();
        let full = fs::metadata(&path).is_ok_and(|log| log.len() >= Self::ERROR_LOG_LIMIT);
        let mut options = OpenOptions::new();
        if full {
            options.create(true).write(true).truncate(true);
        } else {
            options.create(true).append(true);
        }
        let message = format!("{error:#}").replace(['\r', '\n'], " ");
        let line = format!("{} {message}\n", Timestamp::now());
        if let Ok(mut log) = options.open(&path) {
            let _ = log.write_all(line.as_bytes());
        }
    }

    pub fn open_store(&self, patience: Patience) -> Result<Store> {
        Store::open(&self.database(), patience)
    }

    pub fn embedder(&self) -> Option<Embedder> {
        self.open_embedder().ok().flatten()
    }

    pub fn semantic(&self) -> Option<(Embedder, VectorIndex)> {
        self.open_semantic().ok().flatten()
    }

    pub fn open_embedder(&self) -> Result<Option<Embedder>> {
        let path = self.embeddings();
        if !path.exists() {
            return Ok(None);
        }
        Embedder::open(&path)
            .context("load the embedding model")
            .map(Some)
    }

    pub fn open_index(&self) -> Result<Option<VectorIndex>> {
        let path = self.index();
        if !path.exists() {
            return Ok(None);
        }
        VectorIndex::open(&path)
            .context("load the recall index")
            .map(Some)
    }

    pub fn open_semantic(&self) -> Result<Option<(Embedder, VectorIndex)>> {
        let Some(embedder) = self.open_embedder()? else {
            return Ok(None);
        };
        Ok(self.open_index()?.map(|index| (embedder, index)))
    }

    pub fn recall<'a>(&self, store: &'a Store) -> Recall<'a> {
        match self.open_semantic() {
            Ok(semantic) => Recall::new(store, semantic),
            Err(error) => Recall::new(store, None).semantic_unavailable(&error),
        }
    }
}

#[cfg(test)]
mod tests {
    use trodden_embed::{DIMS, QUANTIZED_LEN};
    use trodden_recall::{Abstention, Decision, Query};

    use super::*;

    struct Scratch {
        home: Home,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-home-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let home = Home::at(dir);
            home.initialize().expect("home initializes");
            Self { home }
        }

        fn write(path: &Path, bytes: &[u8]) {
            fs::write(path, bytes).expect("scratch file is writable");
        }

        fn install_pack(&self) {
            let token = b"crash";
            let mut pack = b"TRDEMB\x01\0".to_vec();
            for field in [DIMS, 1, token.len(), 0, 0, token.len(), 0] {
                pack.extend(
                    u32::try_from(field)
                        .expect("pack field is small")
                        .to_le_bytes(),
                );
            }
            pack.extend(token);
            pack.extend(1.0_f32.to_le_bytes());
            pack.resize(pack.len() + QUANTIZED_LEN - 4, 1);
            Self::write(&self.home.embeddings(), &pack);
        }

        fn semantic_error(&self) -> Option<String> {
            let store = self.home.open_store(Patience::Batch).expect("store opens");
            let outcome = self
                .home
                .recall(&store)
                .recall(&Query {
                    prompt: "Fix the crash in src/paginate.js",
                    repo: "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c",
                    root: Path::new("."),
                    session: None,
                })
                .expect("recall succeeds");
            assert_eq!(
                outcome.decision,
                Decision::Abstain(Abstention::NoCandidates)
            );
            outcome.semantic_error
        }

        fn logged(&self) -> Vec<String> {
            fs::read_to_string(self.home.error_log())
                .expect("error log reads")
                .lines()
                .map(|line| {
                    let (at, message) = line.split_once(' ').expect("line has a timestamp");
                    at.parse::<Timestamp>()
                        .expect("line starts with a timestamp");
                    message.to_owned()
                })
                .collect()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.home.dir());
        }
    }

    #[test]
    fn missing_semantic_files_leave_semantic_matching_off_without_an_error() {
        let scratch = Scratch::new("semantic-missing");

        assert!(
            scratch
                .home
                .open_semantic()
                .expect("nothing to load")
                .is_none()
        );
        assert_eq!(scratch.semantic_error(), None);

        scratch.install_pack();

        assert!(scratch.home.open_embedder().expect("pack loads").is_some());
        assert!(scratch.home.open_semantic().expect("pack loads").is_none());
        assert_eq!(scratch.semantic_error(), None);
    }

    #[test]
    fn loads_a_valid_pack_and_index() {
        let scratch = Scratch::new("semantic-valid");
        scratch.install_pack();
        VectorIndex::build(&[], &scratch.home.index()).expect("index builds");

        assert!(scratch.home.open_semantic().expect("files load").is_some());
        assert_eq!(scratch.semantic_error(), None);
    }

    #[test]
    fn reports_a_corrupt_pack_and_recalls_without_it() {
        let scratch = Scratch::new("semantic-corrupt-pack");
        Scratch::write(&scratch.home.embeddings(), b"TRDEMB\x01\0short");

        let error = scratch
            .home
            .open_semantic()
            .expect_err("a corrupt pack is reported");

        assert!(format!("{error:#}").starts_with("load the embedding model: "));
        assert!(scratch.home.semantic().is_none());
        let reported = scratch.semantic_error().expect("recall notes the error");
        assert_eq!(reported, format!("{error:#}"));
    }

    #[test]
    fn reports_a_corrupt_index_and_recalls_without_it() {
        let scratch = Scratch::new("semantic-corrupt-index");
        scratch.install_pack();
        Scratch::write(&scratch.home.index(), b"not an index at all");

        let error = scratch
            .home
            .open_semantic()
            .expect_err("a corrupt index is reported");

        assert!(format!("{error:#}").starts_with("load the recall index: "));
        let reported = scratch.semantic_error().expect("recall notes the error");
        assert_eq!(reported, format!("{error:#}"));
    }

    #[test]
    fn errors_are_logged_one_per_line_with_their_causes() {
        let scratch = Scratch::new("log-lines");
        let error = anyhow::anyhow!("line 3 is not JSON\nexpected value")
            .context("parse the transcript")
            .context("ingest /home/dev/.claude/projects/shop/session.jsonl");

        scratch.home.log_error(&error);
        scratch.home.log_error("start a background ingest");

        assert_eq!(
            scratch.logged(),
            [
                "ingest /home/dev/.claude/projects/shop/session.jsonl: parse the transcript: line 3 is not JSON expected value",
                "start a background ingest",
            ]
        );
    }

    #[test]
    fn an_uninitialized_home_logs_nothing() {
        let scratch = Scratch::new("log-uninitialized");
        let elsewhere = Home::at(scratch.home.dir().join("elsewhere"));

        elsewhere.log_error("parse the hook payload");

        assert!(!elsewhere.dir().exists());
    }

    #[test]
    fn a_full_error_log_starts_over() {
        let scratch = Scratch::new("log-full");
        let old = "2026-10-01T10:00:00Z parse the hook payload\n";
        let lines = usize::try_from(Home::ERROR_LOG_LIMIT).expect("limit fits") / old.len() + 1;
        Scratch::write(&scratch.home.error_log(), old.repeat(lines).as_bytes());

        scratch.home.log_error("start a background ingest");

        assert_eq!(scratch.logged(), ["start a background ingest"]);
    }

    #[test]
    fn journals_are_named_after_plain_session_ids_only() {
        let home = Home::at("/data/trodden");

        assert_eq!(
            home.journal("cursor", "4f1c2a9e-77d1-4c4b-9a0e-3b1f2c7d8e90"),
            PathBuf::from(
                "/data/trodden/journals/cursor/4f1c2a9e-77d1-4c4b-9a0e-3b1f2c7d8e90.jsonl"
            )
        );
        for hostile in ["../../escape", "", ".hidden", "a/b"] {
            let path = home.journal("cursor", hostile);
            assert_eq!(
                path.parent(),
                Some(Path::new("/data/trodden/journals/cursor"))
            );
            assert!(
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("session-")),
                "{}",
                path.display()
            );
        }
    }
}
