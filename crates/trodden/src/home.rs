use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use trodden_embed::Embedder;
use trodden_recall::VectorIndex;
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

    pub fn hook_log(&self) -> PathBuf {
        self.dir.join("hook.log")
    }

    pub fn open_store(&self, patience: Patience) -> Result<Store> {
        Store::open(&self.database(), patience)
    }

    pub fn embedder(&self) -> Option<Embedder> {
        Embedder::open(&self.embeddings()).ok()
    }

    pub fn semantic(&self) -> Option<(Embedder, VectorIndex)> {
        let embedder = self.embedder()?;
        let index = VectorIndex::open(&self.index()).ok()?;
        Some((embedder, index))
    }
}
