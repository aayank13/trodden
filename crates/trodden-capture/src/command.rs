use trodden_core::trace::ToolAction;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Command {
    text: String,
}

impl Command {
    const SUBCOMMAND_PROGRAMS: &[&str] = &[
        "bun", "bundle", "cargo", "deno", "dotnet", "go", "gradle", "just", "make", "mix", "mvn",
        "npm", "npx", "pnpm", "poetry", "uv", "yarn", "git", "docker", "kubectl",
    ];

    const READ_PROGRAMS: &[&str] = &[
        "cat", "head", "tail", "less", "more", "bat", "wc", "file", "stat",
    ];

    const SEARCH_PROGRAMS: &[&str] = &[
        "ls", "find", "fd", "tree", "grep", "rg", "ag", "ack", "which",
    ];

    pub fn normalize(command: &str, cwd: &str) -> Self {
        let mut text = command.trim().to_owned();
        let cwd = cwd.trim_end_matches('/');
        if cwd.len() > 1 {
            text = text.replace(&format!("{cwd}/"), "./").replace(cwd, ".");
        }
        for prefix in ["cd . && ", "cd . ; ", "cd .; "] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim_start().to_owned();
            }
        }
        Self { text }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn program(&self) -> String {
        let mut words = self
            .first_segment()
            .split_whitespace()
            .skip_while(|word| word.contains('=') || *word == "sudo" || *word == "env");
        let Some(program) = words.next() else {
            return String::new();
        };
        let program = program.rsplit('/').next().unwrap_or(program);
        if Self::SUBCOMMAND_PROGRAMS.contains(&program)
            && let Some(sub) = words.find(|word| !word.starts_with('-'))
        {
            return format!("{program} {sub}");
        }
        if program.starts_with("python") {
            let rest: Vec<&str> = words.collect();
            return match rest.as_slice() {
                ["-m", module, ..] => format!("python -m {module}"),
                [script, ..] => format!("python {script}"),
                [] => "python".to_owned(),
            };
        }
        program.to_owned()
    }

    pub fn action(&self) -> ToolAction {
        let program = self.program();
        let name = program.split_whitespace().next().unwrap_or_default();
        if Self::READ_PROGRAMS.contains(&name) {
            ToolAction::Read
        } else if Self::SEARCH_PROGRAMS.contains(&name) || program == "git grep" {
            ToolAction::Search
        } else {
            ToolAction::Run
        }
    }

    fn first_segment(&self) -> &str {
        self.text
            .split(['|', ';'])
            .next()
            .and_then(|segment| segment.split("&&").next())
            .unwrap_or(&self.text)
            .trim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifies_programs() {
        let cases = [
            ("cargo test -p invoicer", "cargo test"),
            ("RUST_LOG=debug cargo --locked build", "cargo build"),
            ("npm run gen", "npm run"),
            (
                "python3 -m unittest discover -s tests",
                "python -m unittest",
            ),
            (
                "python3 tools/gen_models.py --check",
                "python tools/gen_models.py",
            ),
            ("/usr/bin/grep -rn paginate src | head", "grep"),
            ("sh scripts/test.sh", "sh"),
        ];
        for (command, program) in cases {
            assert_eq!(
                Command::normalize(command, "/work").program(),
                program,
                "{command}"
            );
        }
    }

    #[test]
    fn relativizes_to_the_working_directory() {
        let command = Command::normalize(
            "cd /tmp/run-7/catalog && node /tmp/run-7/catalog/scripts/build.js",
            "/tmp/run-7/catalog",
        );

        assert_eq!(command.text(), "node ./scripts/build.js");
    }

    #[test]
    fn classifies_inspection_commands() {
        assert_eq!(
            Command::normalize("cat src/lib.rs", "/w").action(),
            ToolAction::Read
        );
        assert_eq!(
            Command::normalize("rg -n TODO", "/w").action(),
            ToolAction::Search
        );
        assert_eq!(
            Command::normalize("npm test", "/w").action(),
            ToolAction::Run
        );
    }
}
