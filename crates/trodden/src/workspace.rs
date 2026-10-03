use std::{
    fs,
    path::{self, Path, PathBuf},
    process::Command,
    time::UNIX_EPOCH,
};

use anyhow::{Context, Result};
use trodden_core::RepoId;
use trodden_store::Store;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Workspace {
    pub root: PathBuf,
    pub repo: RepoId,
}

impl Workspace {
    pub fn resolve(cwd: &Path, store: &Store) -> Result<Self> {
        Self::identify(cwd, store, true)
    }

    pub fn resolve_read_only(cwd: &Path, store: &Store) -> Result<Self> {
        Self::identify(cwd, store, false)
    }

    fn identify(cwd: &Path, store: &Store, remember: bool) -> Result<Self> {
        let cwd = path::absolute(cwd)
            .with_context(|| format!("resolve the working directory {:?}", cwd.display()))?;
        let root = Self::find_root(&cwd);
        let repo = match Self::stamp(&root) {
            Some(stamp) => Self::cached_id(&root, &stamp, store, remember)?,
            None => Self::path_id(&root),
        };
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

    fn cached_id(root: &Path, stamp: &str, store: &Store, remember: bool) -> Result<String> {
        let key = format!("{}#{stamp}", root.to_string_lossy());
        if let Some(repo) = store.repo_for_root(&key)? {
            return Ok(repo);
        }
        if Self::is_unborn(root) {
            return Ok(Self::path_id(root));
        }
        let Some(repo) = Self::history_id(root) else {
            return Ok(Self::path_id(root));
        };
        if remember {
            store.remember_repo(&key, &repo)?;
        }
        Ok(repo)
    }

    fn stamp(root: &Path) -> Option<String> {
        let metadata = fs::metadata(root.join(".git")).ok()?;
        let born = metadata
            .created()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |age| age.as_nanos());
        let shallow =
            Self::git_dirs(root).is_some_and(|(_, common)| common.join("shallow").exists());
        Some(format!(
            "{}:{born}:{}",
            Self::inode(&metadata),
            if shallow { "shallow" } else { "full" }
        ))
    }

    #[cfg(unix)]
    fn inode(metadata: &fs::Metadata) -> u64 {
        use std::os::unix::fs::MetadataExt;
        metadata.ino()
    }

    #[cfg(not(unix))]
    fn inode(_metadata: &fs::Metadata) -> u64 {
        0
    }

    fn is_unborn(root: &Path) -> bool {
        let Some((git_dir, common)) = Self::git_dirs(root) else {
            return false;
        };
        let Ok(head) = fs::read_to_string(git_dir.join("HEAD")) else {
            return false;
        };
        head.trim().strip_prefix("ref: ").is_some_and(|reference| {
            [reference, "packed-refs", "reftable"]
                .iter()
                .all(|name| !common.join(name).exists())
        })
    }

    fn git_dirs(root: &Path) -> Option<(PathBuf, PathBuf)> {
        let dot_git = root.join(".git");
        let git_dir = if dot_git.is_dir() {
            dot_git
        } else {
            let link = fs::read_to_string(&dot_git).ok()?;
            root.join(link.strip_prefix("gitdir:")?.trim())
        };
        let common = match fs::read_to_string(git_dir.join("commondir")) {
            Ok(common) => git_dir.join(common.trim()),
            Err(_) => git_dir.clone(),
        };
        Some((git_dir, common))
    }

    fn history_id(root: &Path) -> Option<String> {
        let root_commit = Self::root_commit(root)?;
        let shallow = Self::git(root, &["rev-parse", "--is-shallow-repository"])?;
        if shallow.trim() == "true" {
            Self::remote_id(root)
        } else {
            Some(root_commit)
        }
    }

    fn root_commit(root: &Path) -> Option<String> {
        let out = Self::git(root, &["rev-list", "--max-parents=0", "HEAD"])?;
        out.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .min()
            .map(str::to_owned)
    }

    fn remote_id(root: &Path) -> Option<String> {
        let out = Self::git(root, &["config", "--get-regexp", r"^remote\..*\.url$"])?;
        let remotes: Vec<_> = out
            .lines()
            .filter_map(|line| line.split_once(' '))
            .collect();
        let (_, url) = remotes
            .iter()
            .find(|(key, _)| *key == "remote.origin.url")
            .or_else(|| remotes.first())?;
        Some(format!(
            "remote-{:016x}",
            Self::fnv(&Self::normalize_remote(url))
        ))
    }

    fn normalize_remote(url: &str) -> String {
        let url = url.trim();
        let (authority, path) = match url.split_once("://") {
            Some((_, rest)) => rest.split_once('/').unwrap_or((rest, "")),
            None => match url.split_once(':') {
                Some((host, path)) if !host.contains('/') => (host, path),
                _ => ("", url),
            },
        };
        let host = authority
            .rsplit_once('@')
            .map_or(authority, |(_, host)| host);
        let host = host.split_once(':').map_or(host, |(host, _)| host);
        let path = path.trim_matches('/');
        let path = path
            .strip_suffix(".git")
            .unwrap_or(path)
            .trim_end_matches('/');
        format!("{}/{path}", host.to_ascii_lowercase())
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
        format!("path-{:016x}", Self::fnv(&root.to_string_lossy()))
    }

    fn fnv(text: &str) -> u64 {
        text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Scratch {
        dir: PathBuf,
        store: Store,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("trodden-workspace-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch directory is writable");
            let store = Store::open_in_memory().expect("in-memory store opens");
            Self { dir, store }
        }

        fn repo(&self, name: &str, commits: &[&str]) -> PathBuf {
            let repo = self.dir.join(name);
            fs::create_dir_all(&repo).expect("repository directory is writable");
            git(&repo, &["init", "-q"]);
            for message in commits {
                commit(&repo, message);
            }
            repo
        }

        fn resolve(&self, cwd: &Path) -> String {
            Workspace::resolve(cwd, &self.store)
                .expect("workspace resolves")
                .repo
                .as_str()
                .to_owned()
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn git(dir: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=Dev",
                "-c",
                "user.email=dev@example.com",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            output.status.success(),
            "git {args:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_owned()
    }

    fn commit(repo: &Path, message: &str) {
        fs::write(repo.join("CHANGES"), message).expect("file is writable");
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-q", "-m", message]);
    }

    fn root_commit(repo: &Path) -> String {
        git(repo, &["rev-list", "--max-parents=0", "HEAD"])
    }

    fn text(dir: &Path) -> String {
        dir.to_string_lossy().into_owned()
    }

    #[test]
    fn relative_directories_resolve_to_absolute_roots() {
        let store = Store::open_in_memory().expect("in-memory store opens");
        let here = std::env::current_dir().expect("current directory exists");
        let relative = Workspace::resolve(Path::new("."), &store).expect("relative resolves");
        let absolute = Workspace::resolve(&here, &store).expect("absolute resolves");
        assert!(relative.root.is_absolute());
        assert_eq!(relative, absolute);
        assert!(Workspace::resolve(Path::new(""), &store).is_err());
    }

    #[test]
    fn full_clones_are_identified_by_their_root_commit() {
        let scratch = Scratch::new("full");
        let repo = scratch.repo("shop", &["first", "second"]);
        fs::create_dir_all(repo.join("src")).expect("source directory is writable");
        assert_eq!(scratch.resolve(&repo.join("src")), root_commit(&repo));
        assert_eq!(scratch.resolve(&repo), root_commit(&repo));
    }

    #[test]
    fn a_replaced_clone_gets_its_own_id() {
        let scratch = Scratch::new("replaced");
        let first = scratch.repo("shop", &["shop"]);
        let old = scratch.resolve(&first);
        fs::remove_dir_all(&first).expect("repository is removable");
        let second = scratch.repo("shop", &["blog"]);
        let new = scratch.resolve(&second);
        assert_ne!(old, new);
        assert_eq!(new, root_commit(&second));
    }

    #[test]
    fn a_repository_without_commits_is_identified_once_it_has_one() {
        let scratch = Scratch::new("unborn");
        let repo = scratch.repo("shop", &[]);
        assert!(scratch.resolve(&repo).starts_with("path-"));
        commit(&repo, "first");
        assert_eq!(scratch.resolve(&repo), root_commit(&repo));
    }

    #[test]
    fn shallow_clones_keep_one_id_until_unshallowed() {
        let scratch = Scratch::new("shallow");
        let origin = scratch.repo("origin", &["first", "second", "third"]);
        let url = format!("file://{}", origin.display());
        let full = scratch.dir.join("full");
        let shallow = scratch.dir.join("shallow");
        let worktree = scratch.dir.join("worktree");
        git(&scratch.dir, &["clone", "-q", &url, &text(&full)]);
        git(
            &scratch.dir,
            &["clone", "-q", "--depth", "1", &url, &text(&shallow)],
        );
        git(&shallow, &["worktree", "add", "-q", &text(&worktree)]);

        let shallow_id = scratch.resolve(&shallow);
        assert!(shallow_id.starts_with("remote-"));
        assert_eq!(scratch.resolve(&worktree), shallow_id);
        git(&shallow, &["fetch", "-q", "--deepen", "1"]);
        assert_eq!(scratch.resolve(&shallow), shallow_id);

        git(&shallow, &["fetch", "-q", "--unshallow"]);
        assert_eq!(scratch.resolve(&shallow), root_commit(&origin));
        assert_eq!(scratch.resolve(&worktree), root_commit(&origin));
        assert_eq!(scratch.resolve(&full), root_commit(&origin));
    }

    #[test]
    fn read_only_resolution_uses_the_cache_without_filling_it() {
        let scratch = Scratch::new("read-only");
        let repo = scratch.repo("shop", &["first"]);
        let stamp = Workspace::stamp(&repo).expect("repository has a stamp");
        let key = format!("{}#{stamp}", text(&repo));

        let read_only =
            Workspace::resolve_read_only(&repo, &scratch.store).expect("workspace resolves");
        assert_eq!(read_only.repo.as_str(), root_commit(&repo));
        assert_eq!(
            scratch.store.repo_for_root(&key).expect("cache reads"),
            None
        );

        scratch
            .store
            .remember_repo(&key, "cached")
            .expect("cache writes");
        let cached =
            Workspace::resolve_read_only(&repo, &scratch.store).expect("workspace resolves");
        assert_eq!(cached.repo.as_str(), "cached");
    }

    #[test]
    fn remote_urls_normalize_across_protocols() {
        let urls = [
            "git@github.com:Acme/Shop.git",
            "https://github.com/Acme/Shop",
            "https://token@GitHub.com/Acme/Shop.git/",
            "ssh://git@github.com:22/Acme/Shop.git",
        ];
        for url in urls {
            assert_eq!(Workspace::normalize_remote(url), "github.com/Acme/Shop");
        }
        assert_eq!(
            Workspace::normalize_remote("file:///srv/git/shop.git"),
            Workspace::normalize_remote("/srv/git/shop"),
        );
    }
}
