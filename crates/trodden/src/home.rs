use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use trodden_embed::Embedder;
use trodden_recall::{Recall, VectorIndex};
use trodden_store::{Patience, Store};

#[derive(Debug, Clone)]
pub struct Home {
    dir: PathBuf,
}

impl Home {
    pub const ENV: &str = "TRODDEN_HOME";

    pub fn locate() -> Result<Self> {
        if let Some(dir) = env::var_os(Self::ENV).filter(|dir| !dir.is_empty()) {
            return Ok(Self {
                dir: PathBuf::from(dir),
            });
        }
        let base = if cfg!(windows) {
            env::var_os("LOCALAPPDATA").map(PathBuf::from)
        } else if cfg!(target_os = "macos") {
            env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Application Support"))
        } else {
            env::var_os("XDG_DATA_HOME")
                .filter(|dir| !dir.is_empty())
                .map(PathBuf::from)
                .or_else(|| {
                    env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/share"))
                })
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

    pub fn hook_log(&self) -> PathBuf {
        self.dir.join("hook.log")
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
}
