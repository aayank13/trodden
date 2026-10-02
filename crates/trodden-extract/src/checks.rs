use std::{cmp::Reverse, fs, path::Path, sync::LazyLock};

use regex::Regex;
use trodden_capture::Command;

use crate::{lint::DangerLint, verify::Verification};

const DOCS: &[&str] = &[
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    ".github/CONTRIBUTING.md",
    "docs/CONTRIBUTING.md",
    "README.md",
    "README",
];

const SCRIPTS: &[(&str, &str)] = &[
    ("tools/check.py", "python3 tools/check.py"),
    ("scripts/check.py", "python3 scripts/check.py"),
    ("scripts/check.sh", "sh scripts/check.sh"),
    ("scripts/ci.sh", "sh scripts/ci.sh"),
    ("bin/check", "./bin/check"),
    ("script/check", "./script/check"),
];

static FULL_CHECK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[\s/_.:-])(?:check|checks|ci|verify|validate|precommit|pre-commit)(?:$|[\s/_.:-])")
        .expect("full check pattern is valid")
});

static TARGET: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(check|ci|verify|test|tests)\s*:(?:[^=]|$)").expect("target pattern is valid")
});

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredCheck {
    pub command: String,
    pub source: String,
    strength: Strength,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Strength {
    Partial,
    Tests,
    Full,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProjectChecks {
    checks: Vec<DeclaredCheck>,
}

impl ProjectChecks {
    pub fn discover(root: &Path) -> Self {
        let read = |name: &str| fs::read_to_string(root.join(name)).ok();
        let mut found = Vec::new();

        for doc in DOCS {
            if let Some(text) = read(doc) {
                for command in Self::doc_commands(&text) {
                    found.push((command, (*doc).to_owned()));
                }
            }
        }
        if let Some(text) = read("package.json") {
            found.extend(
                Self::package_scripts(root, &text)
                    .into_iter()
                    .map(|command| (command, "package.json".to_owned())),
            );
        }
        for (file, runner) in [
            ("Makefile", "make"),
            ("justfile", "just"),
            ("Justfile", "just"),
        ] {
            if let Some(text) = read(file) {
                for target in TARGET.captures_iter(&text) {
                    found.push((format!("{runner} {}", &target[1]), file.to_owned()));
                }
            }
        }
        for (file, command) in SCRIPTS {
            if root.join(file).is_file() {
                found.push(((*command).to_owned(), (*file).to_owned()));
            }
        }
        if root.join("Cargo.toml").is_file() {
            found.push(("cargo test".to_owned(), "Cargo.toml".to_owned()));
        }
        if root.join("go.mod").is_file() {
            found.push(("go test ./...".to_owned(), "go.mod".to_owned()));
        }

        Self::from_commands(found)
    }

    pub fn from_commands(commands: impl IntoIterator<Item = (String, String)>) -> Self {
        let mut checks: Vec<DeclaredCheck> = Vec::new();
        for (command, source) in commands {
            let command = command.trim().to_owned();
            if !Verification::is_verify(&command)
                || DangerLint::check(&command).is_some()
                || checks
                    .iter()
                    .any(|check| Self::same(&check.command, &command))
            {
                continue;
            }
            let strength = Self::strength(&command);
            checks.push(DeclaredCheck {
                command,
                source,
                strength,
            });
        }
        checks.sort_by_key(|check| Reverse(check.strength));
        Self { checks }
    }

    pub fn strongest(&self) -> Option<&DeclaredCheck> {
        self.checks.first()
    }

    pub fn find(&self, command: &str) -> Option<&DeclaredCheck> {
        self.checks
            .iter()
            .find(|check| Self::same(&check.command, command))
    }

    pub fn redacted(mut self, redact: impl Fn(&str) -> String) -> Self {
        self.checks
            .retain(|check| redact(&check.command) == check.command);
        self
    }

    pub fn same(a: &str, b: &str) -> bool {
        let words = |command: &str| -> Vec<String> {
            let clean = Verification::clean(command);
            let normalized = Command::normalize(&clean, "");
            normalized
                .text()
                .split_whitespace()
                .map(|word| if word == "python3" { "python" } else { word }.to_owned())
                .collect()
        };
        words(a) == words(b)
    }

    fn strength(command: &str) -> Strength {
        let program = Command::normalize(command, "").program();
        let target = command
            .split_whitespace()
            .skip(1)
            .collect::<Vec<_>>()
            .join(" ");
        let lint_only = target
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| word == "lint");
        if FULL_CHECK.is_match(&program) || FULL_CHECK.is_match(&target) {
            Strength::Full
        } else if lint_only
            || [
                "cargo build",
                "cargo check",
                "cargo clippy",
                "go build",
                "go vet",
                "dotnet build",
                "gradle build",
                "tsc",
                "eslint",
                "ruff",
                "mypy",
                "python -m mypy",
            ]
            .contains(&program.as_str())
        {
            Strength::Partial
        } else {
            Strength::Tests
        }
    }

    fn doc_commands(text: &str) -> Vec<String> {
        static INLINE: LazyLock<Regex> =
            LazyLock::new(|| Regex::new(r"`([^`\n]+)`").expect("inline code pattern is valid"));
        let mut commands = Vec::new();
        let mut fenced = false;
        for line in text.lines() {
            if line.trim_start().starts_with("```") {
                fenced = !fenced;
                continue;
            }
            let code = if fenced {
                Some(line.trim())
            } else {
                line.strip_prefix("    ")
                    .or_else(|| line.strip_prefix('\t'))
                    .map(str::trim)
            };
            if let Some(code) = code {
                commands.push(code.trim_start_matches("$ ").to_owned());
            } else {
                commands.extend(INLINE.captures_iter(line).map(|span| span[1].to_owned()));
            }
        }
        commands
    }

    fn package_scripts(root: &Path, text: &str) -> Vec<String> {
        static SCRIPT: LazyLock<Regex> = LazyLock::new(|| {
            Regex::new(r#""(test|check|ci|verify|validate|lint)"\s*:"#)
                .expect("script pattern is valid")
        });
        let manager = [
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("bun.lock", "bun"),
            ("bun.lockb", "bun"),
        ]
        .iter()
        .find(|(lockfile, _)| root.join(lockfile).is_file())
        .map_or("npm", |(_, manager)| manager);
        let scripts = text
            .split_once("\"scripts\"")
            .and_then(|(_, rest)| rest.split_once('}'))
            .map_or("", |(scripts, _)| scripts);
        SCRIPT
            .captures_iter(scripts)
            .map(|script| match &script[1] {
                "test" => format!("{manager} test"),
                name => format!("{manager} run {name}"),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(name: &str, files: &[(&str, &str)]) -> ProjectChecks {
        let root =
            std::env::temp_dir().join(format!("trodden-checks-{name}-{}", std::process::id()));
        fs::create_dir_all(&root).expect("scratch directory is creatable");
        for (path, contents) in files {
            fs::write(root.join(path), contents).expect("scratch file is writable");
        }
        let checks = ProjectChecks::discover(&root);
        fs::remove_dir_all(&root).expect("scratch directory is removable");
        checks
    }

    #[test]
    fn finds_the_documented_full_check() {
        let checks = project(
            "documented",
            &[(
                "README.md",
                "# tasks\n\nBefore finishing a change, run the project checks:\n\n    python3 tools/check.py\n",
            )],
        );
        let strongest = checks.strongest().expect("the README declares a check");

        assert_eq!(strongest.command, "python3 tools/check.py");
        assert_eq!(strongest.source, "README.md");
        assert!(checks.find("python tools/check.py").is_some());
    }

    #[test]
    fn falls_back_to_scripts_and_ecosystem_defaults() {
        let strongest = |checks: &ProjectChecks| {
            checks
                .strongest()
                .map(|check| (check.command.clone(), check.source.clone()))
        };
        let scripts = project(
            "scripts",
            &[(
                "package.json",
                r#"{ "scripts": { "gen": "node scripts/gen-keys.js", "test": "node --test" } }"#,
            )],
        );
        let cargo = project(
            "cargo",
            &[("Cargo.toml", "[package]\nname = \"invoicer\"\n")],
        );
        let inline = project(
            "inline",
            &[(
                "CONTRIBUTING.md",
                "Run `cargo test` before sending a change.\n",
            )],
        );

        assert_eq!(
            strongest(&scripts),
            Some(("npm test".to_owned(), "package.json".to_owned()))
        );
        assert_eq!(
            strongest(&cargo),
            Some(("cargo test".to_owned(), "Cargo.toml".to_owned()))
        );
        assert_eq!(
            strongest(&inline),
            Some(("cargo test".to_owned(), "CONTRIBUTING.md".to_owned()))
        );
    }

    #[test]
    fn prefers_full_checks_over_tests() {
        let checks = ProjectChecks::from_commands([
            ("cargo build".to_owned(), "README.md".to_owned()),
            ("npm test".to_owned(), "README.md".to_owned()),
            ("make check".to_owned(), "Makefile".to_owned()),
            ("git status".to_owned(), "README.md".to_owned()),
        ]);

        let order: Vec<&str> = checks.checks.iter().map(|c| c.command.as_str()).collect();
        assert_eq!(order, ["make check", "npm test", "cargo build"]);
    }
}
