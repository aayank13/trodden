use std::{
    path::{Path, PathBuf},
    process::Command,
};

use anyhow::Result;
use trodden_core::RepoId;
use trodden_store::Store;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    pub root: PathBuf,
    pub repo: RepoId,
}

impl Workspace {
    pub fn resolve(cwd: &Path, store: &Store) -> Result<Self> {
        let root = Self::find_root(cwd);
        let key = root.to_string_lossy().into_owned();
        if let Some(repo) = store.repo_for_root(&key)? {
            return Ok(Self {
                root,
                repo: RepoId::new(repo),
            });
        }
        let repo = Self::root_commit(&root).unwrap_or_else(|| Self::path_id(&root));
        store.remember_repo(&key, &repo)?;
        Ok(Self {
            root,
            repo: RepoId::new(repo),
        })
    }

    pub fn head(&self) -> Option<String> {
        Self::git(&self.root, &["rev-parse", "HEAD"])
            .and_then(|out| out.lines().next().map(str::to_owned))
    }

    fn find_root(cwd: &Path) -> PathBuf {
        cwd.ancestors()
            .find(|dir| dir.join(".git").exists())
            .unwrap_or(cwd)
            .to_path_buf()
    }

    fn root_commit(root: &Path) -> Option<String> {
        let out = Self::git(root, &["rev-list", "--max-parents=0", "HEAD"])?;
        out.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .min()
            .map(str::to_owned)
    }

    fn git(root: &Path, args: &[&str]) -> Option<String> {
        if !root.join(".git").exists() {
            return None;
        }
        let output = Command::new("git")
            .arg("-C")
            .arg(root)
            .args(args)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    }

    fn path_id(root: &Path) -> String {
        let hash = root
            .to_string_lossy()
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
            });
        format!("path-{hash:016x}")
    }
}
