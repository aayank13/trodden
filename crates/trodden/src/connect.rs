use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process,
};

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    path: PathBuf,
}

impl Program {
    pub fn current() -> Result<Self> {
        Ok(Self::at(
            env::current_exe().context("find the trodden executable")?,
        ))
    }

    pub fn at(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    // Absolute path: agents started from a GUI or a login shell often lack `~/.cargo/bin` on
    // PATH.
    pub fn hook(&self, harness: &str) -> String {
        format!(
            "{} hook {harness}",
            Self::quoted(&self.path.to_string_lossy())
        )
    }

    fn quoted(text: &str) -> String {
        if !text.is_empty()
            && text
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-/+@%,:=".contains(c))
        {
            text.to_owned()
        } else {
            format!("'{}'", text.replace('\'', r"'\''"))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Change {
    pub path: PathBuf,
    pub contents: Option<String>,
    pub executable: bool,
}

impl Change {
    pub fn write(path: impl Into<PathBuf>, contents: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            contents: Some(contents.into()),
            executable: false,
        }
    }

    pub fn remove(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            contents: None,
            executable: false,
        }
    }

    #[must_use]
    pub fn executable(mut self) -> Self {
        self.executable = true;
        self
    }

    pub fn is_needed(&self) -> bool {
        match (&self.contents, fs::read_to_string(&self.path)) {
            (Some(contents), Ok(current)) => *contents != current,
            (Some(_), Err(_)) => true,
            (None, _) => self.path.exists(),
        }
    }

    // Replaced in one rename so the agent never reads half a file; symlinks are written through
    // so dotfile links survive.
    pub fn apply(&self) -> Result<()> {
        let is_link =
            fs::symlink_metadata(&self.path).is_ok_and(|meta| meta.file_type().is_symlink());
        if is_link && self.contents.is_some() {
            let target = fs::canonicalize(&self.path)
                .with_context(|| format!("follow the link {}", self.path.display()))?;
            return Self {
                path: target,
                ..self.clone()
            }
            .apply();
        }
        let Some(contents) = &self.contents else {
            return match fs::remove_file(&self.path) {
                Err(error) if error.kind() != ErrorKind::NotFound => {
                    Err(error).with_context(|| format!("remove {}", self.path.display()))
                }
                _ => Ok(()),
            };
        };
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        let staged = self
            .path
            .with_extension(format!("trodden-{}.tmp", process::id()));
        fs::write(&staged, contents).with_context(|| format!("write {}", staged.display()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = if self.executable {
                0o755
            } else {
                fs::metadata(&self.path).map_or(0o644, |metadata| metadata.permissions().mode())
            };
            fs::set_permissions(&staged, fs::Permissions::from_mode(mode))
                .with_context(|| format!("set the permissions of {}", staged.display()))?;
        }
        fs::rename(&staged, &self.path).with_context(|| format!("replace {}", self.path.display()))
    }
}

// Only handlers whose command is `trodden ... hook <agent>` are ever added or removed.
#[derive(Debug, Clone)]
pub(crate) struct HookFile {
    path: PathBuf,
    original: Option<String>,
    value: Value,
}

impl HookFile {
    pub(crate) fn load(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let original = match fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(error) if error.kind() == ErrorKind::NotFound => None,
            Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
        };
        let value = match original.as_deref().map(str::trim) {
            None | Some("") => Value::Object(Map::new()),
            Some(text) => match serde_json::from_str::<Value>(text) {
                Ok(value @ Value::Object(_)) => value,
                _ => bail!(
                    "{} is not a plain JSON object (comments are not supported); add the hooks by hand",
                    path.display()
                ),
            },
        };
        Ok(Self {
            path,
            original,
            value,
        })
    }

    pub(crate) fn is_ours(command: &str, harness: &str) -> bool {
        command.contains("trodden") && command.contains(&format!(" hook {harness}"))
    }

    fn object<'v>(value: &'v mut Value, path: &[&str]) -> Result<&'v mut Map<String, Value>> {
        let mut current = value;
        for key in path {
            let Value::Object(map) = current else {
                bail!("expected `{key}` to be inside a JSON object");
            };
            current = map
                .entry((*key).to_owned())
                .or_insert_with(|| Value::Object(Map::new()));
        }
        match current {
            Value::Object(map) => Ok(map),
            _ => bail!("expected `{}` to be a JSON object", path.join(".")),
        }
    }

    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        self.value.get(key)
    }

    pub(crate) fn set(&mut self, key: &str, value: Value) -> Result<()> {
        Self::object(&mut self.value, &[])?.insert(key.to_owned(), value);
        Ok(())
    }

    pub(crate) fn add(&mut self, root: &[&str], event: &str, entry: Value) -> Result<()> {
        let events = Self::object(&mut self.value, root)
            .with_context(|| format!("add a hook to {}", self.path.display()))?;
        let entries = events
            .entry(event.to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        let Value::Array(entries) = entries else {
            bail!(
                "expected the `{event}` hooks in {} to be a list",
                self.path.display()
            );
        };
        entries.push(entry);
        Ok(())
    }

    pub(crate) fn remove(&mut self, root: &[&str], harness: &str) -> bool {
        let mut current = &mut self.value;
        for key in root {
            match current.get_mut(*key) {
                Some(next) => current = next,
                None => return false,
            }
        }
        let Value::Object(events) = current else {
            return false;
        };
        let mut removed = false;
        for entries in events.values_mut() {
            let Value::Array(entries) = entries else {
                continue;
            };
            entries.retain_mut(|entry| {
                if Self::handler_is_ours(entry, harness) {
                    removed = true;
                    return false;
                }
                let Some(Value::Array(handlers)) = entry.get_mut("hooks") else {
                    return true;
                };
                let before = handlers.len();
                handlers.retain(|handler| !Self::handler_is_ours(handler, harness));
                removed |= handlers.len() != before;
                !(handlers.is_empty() && before > 0)
            });
        }
        events.retain(|_, entries| !matches!(entries, Value::Array(list) if list.is_empty()));
        if removed && events.is_empty() {
            self.prune(root);
        }
        removed
    }

    fn prune(&mut self, root: &[&str]) {
        let Some((last, parents)) = root.split_last() else {
            return;
        };
        let mut current = &mut self.value;
        for key in parents {
            match current.get_mut(*key) {
                Some(next) => current = next,
                None => return,
            }
        }
        if let Value::Object(map) = current {
            map.remove(*last);
        }
    }

    pub(crate) fn contains(&self, root: &[&str], harness: &str) -> bool {
        let mut copy = self.clone();
        copy.remove(root, harness)
    }

    fn handler_is_ours(handler: &Value, harness: &str) -> bool {
        ["command", "bash", "powershell"].iter().any(|key| {
            handler
                .get(key)
                .and_then(Value::as_str)
                .is_some_and(|command| Self::is_ours(command, harness))
        })
    }

    pub(crate) fn change(&self) -> Result<Option<Change>> {
        let mut text =
            serde_json::to_string_pretty(&self.value).context("encode the hook settings")?;
        text.push('\n');
        let unchanged = self.original.as_deref().is_some_and(|original| {
            serde_json::from_str::<Value>(original).is_ok_and(|value| value == self.value)
        }) || (self.original.is_none() && self.value == Value::Object(Map::new()));
        Ok((!unchanged).then(|| Change::write(&self.path, text)))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Debug)]
    struct Settings {
        dir: PathBuf,
    }

    impl Settings {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-connect-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn file(&self, contents: &Value) -> PathBuf {
            let path = self.dir.join("settings.json");
            fs::write(&path, contents.to_string()).expect("settings are writable");
            path
        }

        fn read(path: &Path) -> Value {
            serde_json::from_str(&fs::read_to_string(path).expect("settings are readable"))
                .expect("settings are JSON")
        }
    }

    impl Drop for Settings {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn programs_are_quoted_only_when_needed() {
        assert_eq!(
            Program::at("/home/dev/.cargo/bin/trodden").hook("codex"),
            "/home/dev/.cargo/bin/trodden hook codex"
        );
        assert_eq!(
            Program::at("/Users/dev/Application Support/it's/trodden").hook("codex"),
            r"'/Users/dev/Application Support/it'\''s/trodden' hook codex"
        );
    }

    #[test]
    fn our_handlers_are_replaced_and_others_kept() {
        let settings = Settings::new("replace");
        let path = settings.file(&json!({
            "theme": "dark",
            "hooks": {
                "SessionStart": [
                    {"hooks": [{"type": "command", "command": "bash ~/.codex/herdr.sh session"}]},
                    {"hooks": [{"type": "command", "command": "/old/trodden hook codex"}]}
                ],
                "Stop": [{"hooks": [
                    {"type": "command", "command": "notify-send done"},
                    {"type": "command", "command": "/old/trodden hook codex"}
                ]}]
            }
        }));
        let mut file = HookFile::load(&path).expect("settings load");

        assert!(file.remove(&["hooks"], "codex"));
        file.add(
            &["hooks"],
            "Stop",
            json!({"hooks": [{"type": "command", "command": "/new/trodden hook codex"}]}),
        )
        .expect("hook is added");
        file.change()
            .expect("settings encode")
            .expect("settings changed")
            .apply()
            .expect("settings are written");

        assert_eq!(
            Settings::read(&path),
            json!({
                "theme": "dark",
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "bash ~/.codex/herdr.sh session"}]}
                    ],
                    "Stop": [
                        {"hooks": [{"type": "command", "command": "notify-send done"}]},
                        {"hooks": [{"type": "command", "command": "/new/trodden hook codex"}]}
                    ]
                }
            })
        );
    }

    #[test]
    fn flat_handlers_are_removed_too() {
        let settings = Settings::new("flat");
        let path = settings.file(&json!({
            "version": 1,
            "hooks": {"stop": [{"command": "'/opt/trodden' hook cursor"}, {"command": "afplay done.aiff"}]}
        }));
        let mut file = HookFile::load(&path).expect("settings load");

        assert!(file.contains(&["hooks"], "cursor"));
        assert!(!file.contains(&["hooks"], "codex"));
        file.remove(&["hooks"], "cursor");

        assert_eq!(
            file.value,
            json!({"version": 1, "hooks": {"stop": [{"command": "afplay done.aiff"}]}})
        );
    }

    #[test]
    fn an_unchanged_file_needs_no_change() {
        let settings = Settings::new("unchanged");
        let path = settings.file(&json!({"hooks": {}}));
        let file = HookFile::load(&path).expect("settings load");

        assert_eq!(file.change().expect("settings encode"), None);
        let missing = HookFile::load(settings.dir.join("missing.json")).expect("missing loads");
        assert_eq!(missing.change().expect("settings encode"), None);
    }

    #[test]
    fn files_with_comments_are_left_alone() {
        let settings = Settings::new("comments");
        let path = settings.dir.join("settings.json");
        fs::write(&path, "{\n  // theme\n  \"theme\": \"dark\"\n}\n").expect("written");

        let error = HookFile::load(&path).expect_err("comments are refused");

        assert!(
            format!("{error:#}").contains("add the hooks by hand"),
            "{error:#}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_settings_are_written_through_to_their_target() {
        let settings = Settings::new("symlink");
        let target = settings.file(&json!({"theme": "dark"}));
        let link = settings.dir.join("linked.json");
        std::os::unix::fs::symlink(&target, &link).expect("symlink is created");

        Change::write(&link, "{}\n").apply().expect("written");

        assert!(
            fs::symlink_metadata(&link)
                .expect("link exists")
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read_to_string(&target).expect("target reads"), "{}\n");
    }

    #[test]
    fn written_files_keep_their_permissions() {
        let settings = Settings::new("permissions");
        let path = settings.file(&json!({}));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).expect("chmod");
            Change::write(&path, "{}\n").apply().expect("written");
            let mode = fs::metadata(&path).expect("stat").permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }
        Change::remove(&path).apply().expect("removed");
        assert!(!path.exists());
        Change::remove(&path)
            .apply()
            .expect("removing twice is fine");
    }
}
