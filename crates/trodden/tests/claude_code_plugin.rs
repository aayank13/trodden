#![cfg(unix)]

use std::{
    env, fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::{self, Command},
};

use serde_json::Value;

struct Scratch {
    dir: PathBuf,
}

impl Scratch {
    const HOOKS: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../integrations/claude-code/hooks/hooks.json"
    );

    fn new(name: &str) -> Self {
        let dir = env::temp_dir().join(format!("trodden-plugin-{name}-{}", process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("scratch dir is writable");
        Self { dir }
    }

    fn cargo_bin(&self) -> PathBuf {
        self.dir.join("cargo").join("bin")
    }

    fn other_bin(&self) -> PathBuf {
        self.dir.join("other")
    }

    fn install(&self, dir: &Path, name: &str) {
        fs::create_dir_all(dir).expect("bin dir is writable");
        let program = dir.join("trodden");
        fs::write(&program, format!("#!/bin/sh\necho \"{name} $*\"\n"))
            .expect("bin dir is writable");
        fs::set_permissions(&program, fs::Permissions::from_mode(0o755)).expect("program is ours");
    }

    fn commands() -> Vec<String> {
        let hooks: Value = serde_json::from_str(
            &fs::read_to_string(Self::HOOKS).expect("plugin hooks are readable"),
        )
        .expect("plugin hooks are JSON");
        let commands: Vec<String> = hooks["hooks"]
            .as_object()
            .expect("hooks map events to groups")
            .values()
            .flat_map(|groups| groups.as_array().expect("event lists groups"))
            .flat_map(|group| group["hooks"].as_array().expect("group lists hooks"))
            .map(|hook| {
                hook["command"]
                    .as_str()
                    .expect("hook has a command")
                    .to_owned()
            })
            .collect();
        assert!(!commands.is_empty(), "plugin declares hooks");
        commands
    }

    fn run_each(&self, path: &[PathBuf]) -> Vec<String> {
        let path = env::join_paths(
            path.iter()
                .chain(&[PathBuf::from("/usr/bin"), PathBuf::from("/bin")]),
        )
        .expect("scratch paths hold no separator");
        Self::commands()
            .iter()
            .map(|command| {
                let output = Command::new("/bin/sh")
                    .arg("-c")
                    .arg(command)
                    .env_clear()
                    .env("HOME", &self.dir)
                    .env("CARGO_HOME", self.dir.join("cargo"))
                    .env("PATH", &path)
                    .output()
                    .expect("sh runs");
                assert!(output.status.success(), "{command} fails: {output:?}");
                String::from_utf8(output.stdout).expect("fake trodden prints UTF-8")
            })
            .collect()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

#[test]
fn the_cargo_install_wins_when_the_agent_path_lacks_cargo_bin() {
    let scratch = Scratch::new("missing-cargo-bin");
    scratch.install(&scratch.other_bin(), "stale");
    scratch.install(&scratch.cargo_bin(), "cargo");
    for printed in scratch.run_each(&[scratch.other_bin()]) {
        assert_eq!(printed, "cargo hook claude-code\n");
    }
}

#[test]
fn an_agent_path_that_lists_cargo_bin_keeps_its_order() {
    let scratch = Scratch::new("listed-cargo-bin");
    scratch.install(&scratch.other_bin(), "preferred");
    scratch.install(&scratch.cargo_bin(), "cargo");
    for printed in scratch.run_each(&[scratch.other_bin(), scratch.cargo_bin()]) {
        assert_eq!(printed, "preferred hook claude-code\n");
    }
}

#[test]
fn a_trodden_found_only_on_path_runs() {
    let scratch = Scratch::new("path-only");
    scratch.install(&scratch.other_bin(), "prebuilt");
    for printed in scratch.run_each(&[scratch.other_bin()]) {
        assert_eq!(printed, "prebuilt hook claude-code\n");
    }
}

#[test]
fn hooks_stay_quiet_without_trodden() {
    let scratch = Scratch::new("absent");
    for printed in scratch.run_each(&[]) {
        assert_eq!(printed, "");
    }
}
