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

    const MAX_QUOTED_CHARS: usize = 200;

    pub fn normalize(command: &str, cwd: &str) -> Self {
        let mut text = Self::without_bodies(command.trim());
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

    fn without_bodies(command: &str) -> String {
        let chars: Vec<char> = command.chars().collect();
        let mut out = String::with_capacity(command.len());
        let mut heredocs: Vec<(String, bool)> = Vec::new();
        let mut index = 0;
        while index < chars.len() {
            match chars[index] {
                quote @ ('\'' | '"') => {
                    let end = Self::closing_quote(&chars, index);
                    let inner = &chars[index + 1..end];
                    out.push(quote);
                    if inner.contains(&'\n') || inner.len() > Self::MAX_QUOTED_CHARS {
                        out.push_str("...");
                    } else {
                        out.extend(inner);
                    }
                    if end < chars.len() {
                        out.push(quote);
                    }
                    index = end + 1;
                }
                '<' if chars[index..].starts_with(&['<', '<', '<']) => {
                    out.push_str("<<<");
                    index += 3;
                }
                '<' if chars[index..].starts_with(&['<', '<']) => {
                    let (end, heredoc) = Self::heredoc(&chars, index + 2);
                    out.extend(&chars[index..end]);
                    heredocs.extend(heredoc);
                    index = end;
                }
                '\n' if !heredocs.is_empty() => {
                    index = Self::skip_bodies(&chars, index + 1, heredocs.drain(..), &mut out);
                    if index < chars.len() {
                        out.push('\n');
                    }
                }
                '\\' => {
                    out.extend(chars.get(index..index + 2).unwrap_or(&chars[index..]));
                    index += 2;
                }
                c => {
                    out.push(c);
                    index += 1;
                }
            }
        }
        out.trim_end().to_owned()
    }

    fn closing_quote(chars: &[char], open: usize) -> usize {
        let quote = chars[open];
        let mut index = open + 1;
        while index < chars.len() && chars[index] != quote {
            index += if quote == '"' && chars[index] == '\\' {
                2
            } else {
                1
            };
        }
        index.min(chars.len())
    }

    fn heredoc(chars: &[char], mut index: usize) -> (usize, Option<(String, bool)>) {
        let strip_tabs = chars.get(index) == Some(&'-');
        index += usize::from(strip_tabs);
        while chars.get(index).is_some_and(|c| *c == ' ' || *c == '\t') {
            index += 1;
        }
        let quote = chars
            .get(index)
            .copied()
            .filter(|c| matches!(c, '\'' | '"'));
        index += usize::from(quote.is_some());
        let start = index;
        while chars
            .get(index)
            .is_some_and(|&c| !c.is_whitespace() && Some(c) != quote && !";&|<>()".contains(c))
        {
            index += 1;
        }
        let delimiter: String = chars[start..index].iter().collect();
        index += usize::from(quote.is_some() && chars.get(index).copied() == quote);
        (
            index,
            (!delimiter.is_empty()).then_some((delimiter, strip_tabs)),
        )
    }

    fn skip_bodies(
        chars: &[char],
        mut index: usize,
        heredocs: impl Iterator<Item = (String, bool)>,
        out: &mut String,
    ) -> usize {
        for (delimiter, strip_tabs) in heredocs {
            while index < chars.len() {
                let end = chars[index..]
                    .iter()
                    .position(|&c| c == '\n')
                    .map_or(chars.len(), |offset| index + offset);
                let line: String = chars[index..end].iter().collect();
                index = end + 1;
                let closing = if strip_tabs {
                    line.trim_start_matches('\t')
                } else {
                    &line
                };
                if closing == delimiter {
                    out.push('\n');
                    out.push_str(&line);
                    break;
                }
            }
        }
        index.min(chars.len())
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
    fn drops_heredoc_bodies_and_long_quoted_text() {
        let long = "x".repeat(300);
        let cases = [
            (
                "cat > src/config.py <<'EOF'\nAPI_URL = 'https://staging'\nprint('hi')\nEOF",
                "cat > src/config.py <<'EOF'\nEOF",
            ),
            (
                "cat <<EOF > notes.md\nfirst\nsecond\nEOF\nnpm test",
                "cat <<EOF > notes.md\nEOF\nnpm test",
            ),
            (
                "cat <<-END > notes.md\n\tindented\n\tEND",
                "cat <<-END > notes.md\n\tEND",
            ),
            ("cat <<EOF\nnever closed", "cat <<EOF"),
            (
                "python3 -c \"import json\nprint(json.dumps({}))\"",
                "python3 -c \"...\"",
            ),
            (
                &format!("echo '{long}' > data.json"),
                "echo '...' > data.json",
            ),
        ];
        for (command, expected) in cases {
            let normalized = Command::normalize(command, "/work");
            assert_eq!(normalized.text(), expected, "{command}");
            assert_eq!(Command::normalize(expected, "/work").text(), expected);
        }
    }

    #[test]
    fn keeps_short_quoted_text_and_lookalikes() {
        for command in [
            "sh -c \"$(curl -fsSL https://get.example.sh)\"",
            "git commit -m 'fix paging'",
            "cat <<< \"hello\"",
            "echo \"<<EOF\"",
            "cargo test \\\n  --workspace",
            "echo 'it\\'s'",
        ] {
            assert_eq!(Command::normalize(command, "/work").text(), command);
        }
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
