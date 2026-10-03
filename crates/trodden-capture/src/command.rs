use std::mem;

use trodden_core::trace::ToolAction;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Command {
    text: String,
}

impl Command {
    const SUBCOMMAND_PROGRAMS: &[&str] = &[
        "bun", "bundle", "cargo", "deno", "dotnet", "go", "gradle", "just", "make", "mix", "mvn",
        "npm", "pnpm", "poetry", "uv", "yarn", "git", "docker", "kubectl",
    ];

    const WRAPPERS: &[&str] = &["sudo", "env", "time", "nice", "nohup", "timeout"];

    const PACKAGE_RUNNERS: &[&str] = &[
        "npx",
        "bunx",
        "pnpx",
        "uvx",
        "uv run",
        "poetry run",
        "bundle exec",
        "npm exec",
        "pnpm exec",
        "pnpm dlx",
        "yarn exec",
        "yarn dlx",
    ];

    const VALUE_OPTIONS: &[(&str, &str)] = &[
        (
            "sudo",
            "-u --user -g --group -C --close-from -D --chdir -p --prompt",
        ),
        ("env", "-u --unset -C --chdir -S --split-string"),
        ("time", "-f --format -o --output"),
        ("nice", "-n --adjustment"),
        ("timeout", "-s --signal -k --kill-after"),
        ("npx", "-p --package -c --call"),
        ("npm exec", "-p --package -c --call -w --workspace"),
        ("bunx", "-p --package"),
        ("uvx", "--from --with -p --python"),
        (
            "uv run",
            "--with -p --python --package --extra --group --directory --project --env-file",
        ),
        ("pnpm dlx", "--package"),
        ("yarn dlx", "-p --package"),
        ("bun", "--cwd"),
        ("cargo", "--color --config -Z -C"),
        ("docker", "-H --host -c --context --config -l --log-level"),
        (
            "git",
            "-C -c --git-dir --work-tree --namespace --config-env",
        ),
        ("go", "-C"),
        (
            "gradle",
            "-p --project-dir -b --build-file -c --settings-file -I --init-script -x --exclude-task --task",
        ),
        ("just", "-f --justfile -d --working-directory"),
        (
            "kubectl",
            "-n --namespace --context --kubeconfig --cluster --user -s --server --as",
        ),
        (
            "make",
            "-C --directory -f --file -I --include-dir -o --old-file",
        ),
        (
            "mvn",
            "-f --file -pl --projects -P --activate-profiles -s --settings -T --threads -rf --resume-from -D",
        ),
        ("npm", "--prefix -w --workspace --loglevel"),
        ("pnpm", "-C --dir -F --filter"),
        ("poetry", "-C --directory -P --project"),
        ("uv", "--directory --project"),
        ("yarn", "--cwd"),
    ];

    const SHELL_BUILTINS: &[&str] = &["cd", "time"];

    const READ_PROGRAMS: &[&str] = &[
        "cat", "head", "tail", "less", "more", "bat", "wc", "file", "stat",
    ];

    const SEARCH_PROGRAMS: &[&str] = &[
        "ls", "find", "fd", "tree", "grep", "rg", "ag", "ack", "which",
    ];

    const MAX_QUOTED_CHARS: usize = 200;

    const PATH_DELIMITERS: &str = "'\"=;&|()<>`";

    pub fn normalize(command: &str, cwd: &str) -> Self {
        let mut text = Self::without_bodies(command.trim());
        let cwd = cwd.trim_end_matches('/');
        if cwd.len() > 1 {
            text = Self::relativize(&text, cwd);
        }
        for prefix in ["cd . && ", "cd . ; ", "cd .; "] {
            if let Some(rest) = text.strip_prefix(prefix) {
                text = rest.trim_start().to_owned();
            }
        }
        Self { text }
    }

    fn relativize(text: &str, cwd: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut copied = 0;
        for (start, _) in text.match_indices(cwd) {
            let end = start + cwd.len();
            let opens = text[..start]
                .chars()
                .next_back()
                .is_none_or(Self::is_path_delimiter);
            let closes = text[end..]
                .chars()
                .next()
                .is_none_or(|c| matches!(c, '/' | ':') || Self::is_path_delimiter(c));
            if opens && closes {
                out.push_str(&text[copied..start]);
                out.push('.');
                copied = end;
            }
        }
        out.push_str(&text[copied..]);
        out
    }

    fn is_path_delimiter(c: char) -> bool {
        c.is_whitespace() || Self::PATH_DELIMITERS.contains(c)
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
        let argv = self.argv();
        let Some((first, arguments)) = argv.split_first() else {
            return String::new();
        };
        let program = Self::name(first);
        if let Some(index) = Self::subcommand(program, arguments) {
            return format!("{program} {}", arguments[index]);
        }
        if program.starts_with("python") {
            let rest: Vec<&str> = arguments.iter().map(String::as_str).collect();
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

    pub fn argv(&self) -> Vec<String> {
        self.layers().1
    }

    pub fn operands(&self) -> Vec<String> {
        let argv = self.argv();
        let Some((first, arguments)) = argv.split_first() else {
            return Vec::new();
        };
        let program = Self::name(first);
        let mut operands = Vec::new();
        let mut index = Self::operand(program, arguments, 0);
        while let Some(operand) = arguments.get(index) {
            operands.push(operand.clone());
            index = Self::operand(program, arguments, index + 1);
        }
        operands
    }

    pub fn executables(&self) -> Vec<String> {
        let (launchers, argv) = self.layers();
        let mut executables = Vec::new();
        let mut resolves_program = false;
        for (launcher, words) in launchers {
            executables.push(words[0].clone());
            if Self::PACKAGE_RUNNERS.contains(&launcher.as_str()) {
                resolves_program = true;
                break;
            }
        }
        if !resolves_program {
            executables.extend(argv.into_iter().next());
        }
        executables.retain(|executable| !Self::SHELL_BUILTINS.contains(&executable.as_str()));
        executables
    }

    pub fn locate(&self, path: &str) -> Option<String> {
        if path.starts_with('~') {
            return None;
        }
        let mut parts = Vec::new();
        for change in self.first_words().0 {
            match change.as_slice() {
                [_, dir] if !dir.starts_with(['/', '~', '-']) => parts.push(dir.clone()),
                _ => return None,
            }
        }
        parts.push(path.to_owned());
        let parts: Vec<&str> = parts
            .iter()
            .flat_map(|part| part.split('/'))
            .filter(|part| !part.is_empty() && *part != ".")
            .collect();
        Some(parts.join("/"))
    }

    fn layers(&self) -> (Vec<(String, Vec<String>)>, Vec<String>) {
        let mut words = self.first_words().1;
        let mut launchers = Vec::new();
        loop {
            let start = words
                .iter()
                .position(|word| !Self::is_assignment(word))
                .unwrap_or(words.len());
            words.drain(..start);
            let Some((launcher, start)) = Self::launched(&words) else {
                return (launchers, words);
            };
            let program = words.split_off(start);
            launchers.push((launcher, words));
            words = program;
        }
    }

    fn first_words(&self) -> (Vec<Vec<String>>, Vec<String>) {
        let mut changes = Vec::new();
        let mut words = Vec::new();
        let mut word: Option<String> = None;
        let mut quote = None;
        let mut previous = ' ';
        let mut chars = self.text.chars().peekable();
        while let Some(c) = chars.next() {
            match (quote, c) {
                (Some(open), _) if c == open => quote = None,
                (Some('"'), '\\') => word.get_or_insert_default().extend(chars.next()),
                (Some(_), _) => word.get_or_insert_default().push(c),
                (None, '\'' | '"') => {
                    quote = Some(c);
                    word.get_or_insert_default();
                }
                (None, '\\') => {
                    if let Some(next) = chars.next().filter(|next| *next != '\n') {
                        word.get_or_insert_default().push(next);
                    }
                }
                (None, '&') if previous == '>' || chars.peek() == Some(&'>') => {
                    word.get_or_insert_default().push(c);
                }
                (None, '|' | ';' | '&' | '\n') => {
                    words.extend(word.take());
                    let chained = c != '|' && (c != '&' || chars.next_if_eq(&'&').is_some());
                    if !chained || words.first().is_some_and(|first| first != "cd") {
                        return (changes, words);
                    }
                    if !words.is_empty() {
                        changes.push(mem::take(&mut words));
                    }
                }
                (None, _) if c.is_whitespace() => words.extend(word.take()),
                (None, _) => word.get_or_insert_default().push(c),
            }
            previous = c;
        }
        words.extend(word);
        (changes, words)
    }

    fn launched(words: &[String]) -> Option<(String, usize)> {
        let (first, arguments) = words.split_first()?;
        let name = Self::name(first);
        let (launcher, start) = match Self::subcommand(name, arguments) {
            Some(index) => (format!("{name} {}", arguments[index]), index + 2),
            None => (name.to_owned(), 1),
        };
        if !Self::WRAPPERS.contains(&launcher.as_str())
            && !Self::PACKAGE_RUNNERS.contains(&launcher.as_str())
        {
            return None;
        }
        let start = Self::operand(&launcher, words, start);
        (start < words.len()).then_some((launcher, start))
    }

    fn subcommand(program: &str, arguments: &[String]) -> Option<usize> {
        if !Self::SUBCOMMAND_PROGRAMS.contains(&program) {
            return None;
        }
        let index = Self::operand(program, arguments, 0);
        (index < arguments.len()).then_some(index)
    }

    fn operand(program: &str, words: &[String], mut index: usize) -> usize {
        let values = Self::VALUE_OPTIONS
            .iter()
            .find(|(name, _)| *name == program)
            .map_or("", |(_, values)| values);
        while let Some(word) = words.get(index) {
            if word == "--" {
                return index + 1;
            }
            if !word.starts_with(['-', '+']) && !word.starts_with(|c: char| c.is_ascii_digit()) {
                break;
            }
            index += if values.split_whitespace().any(|value| value == word) {
                2
            } else {
                1
            };
        }
        index
    }

    fn name(word: &str) -> &str {
        match word.rsplit('/').next().unwrap_or(word) {
            "gradlew" => "gradle",
            "mvnw" => "mvn",
            name => name,
        }
    }

    fn is_assignment(word: &str) -> bool {
        word.split_once('=').is_some_and(|(name, _)| {
            name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
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
            ("git -C sub status", "git status"),
            ("cargo +nightly test", "cargo test"),
            ("npm --prefix web test", "npm test"),
            ("kubectl -n prod get pods", "kubectl get"),
            ("make -j 8 -C web check", "make check"),
            ("./mvnw -pl core -q verify", "mvn verify"),
            ("./gradlew test", "gradle test"),
            ("sudo -u root make", "make"),
            ("FOO=\"a b\" cargo test", "cargo test"),
            ("time cargo test", "cargo test"),
            (
                "env -u HOME RUST_LOG=info nice -n 5 cargo test",
                "cargo test",
            ),
            ("timeout 300 cargo test", "cargo test"),
            ("cd crates/x && cargo test", "cargo test"),
            ("cd /work/web && npm test", "npm test"),
            ("cd web; cd src\nnpm test", "npm test"),
            ("cd web || npm test", "cd"),
            ("npx jest --ci", "jest"),
            ("npx -p typescript tsc --noEmit", "tsc"),
            ("npm exec -- jest", "jest"),
            ("uv run --with pytest-cov pytest", "pytest"),
            ("uv run python -m pytest", "python -m pytest"),
            ("poetry run pytest", "pytest"),
            ("bundle exec rspec spec/models", "rspec"),
            ("uv sync", "uv sync"),
            ("npx", "npx"),
            ("cargo test 2>&1 | tail -20", "cargo test"),
            ("npm test &> out.log", "npm test"),
            ("", ""),
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
    fn finds_the_arguments_the_program_receives() {
        let cases: [(&str, &[&str]); 5] = [
            ("./bin/check", &["./bin/check"]),
            ("cd web && npx jest --ci", &["jest", "--ci"]),
            (
                "FOO=\"a b\" cargo test -- --list",
                &["cargo", "test", "--", "--list"],
            ),
            (
                "python3 manage.py test 'billing.tests'",
                &["python3", "manage.py", "test", "billing.tests"],
            ),
            (
                "cargo test \\\n  --workspace",
                &["cargo", "test", "--workspace"],
            ),
        ];
        for (command, argv) in cases {
            assert_eq!(
                Command::normalize(command, "/work").argv(),
                argv,
                "{command}"
            );
        }
    }

    #[test]
    fn finds_the_operands_past_options_and_their_values() {
        let cases: [(&str, &[&str]); 7] = [
            ("mvn clean test", &["clean", "test"]),
            (
                "./mvnw -pl core -q clean -DskipTests verify",
                &["clean", "verify"],
            ),
            ("./gradlew clean build -x test", &["clean", "build"]),
            ("cd android && ./gradlew :app:test", &[":app:test"]),
            ("gradle help --task test", &["help"]),
            ("timeout 600 mvn -T 4 install", &["install"]),
            ("", &[]),
        ];
        for (command, operands) in cases {
            assert_eq!(
                Command::normalize(command, "/work").operands(),
                operands,
                "{command}"
            );
        }
    }

    #[test]
    fn lists_the_executables_the_shell_must_find() {
        let cases: [(&str, &[&str]); 14] = [
            ("cargo test", &["cargo"]),
            ("cd web && npm test", &["npm"]),
            ("time cargo test", &["cargo"]),
            ("FOO=\"a b\" cargo test", &["cargo"]),
            ("timeout 600 cargo test", &["timeout", "cargo"]),
            ("env CI=1 nice -n 5 make test", &["env", "nice", "make"]),
            ("sudo -u root /usr/bin/make", &["sudo", "/usr/bin/make"]),
            ("timeout 60 npx jest --ci", &["timeout", "npx"]),
            ("uv run --with pytest-cov pytest", &["uv"]),
            ("npm exec -- jest", &["npm"]),
            ("bundle exec rspec", &["bundle"]),
            ("./gradlew test", &["./gradlew"]),
            ("cd web || npm test", &[]),
            ("", &[]),
        ];
        for (command, executables) in cases {
            assert_eq!(
                Command::normalize(command, "/work").executables(),
                executables,
                "{command}"
            );
        }
    }

    #[test]
    fn locates_paths_from_where_the_command_starts() {
        let cases = [
            ("./gradlew test", "./gradlew", Some("gradlew")),
            (
                "cd android && ./gradlew test",
                "./gradlew",
                Some("android/gradlew"),
            ),
            (
                "cd /work/android && ./gradlew test",
                "./gradlew",
                Some("android/gradlew"),
            ),
            (
                "cd web; cd scripts\n./gen.sh",
                "./gen.sh",
                Some("web/scripts/gen.sh"),
            ),
            (
                "cd .. && tools/check.sh",
                "tools/check.sh",
                Some("../tools/check.sh"),
            ),
            ("cd /opt/android && ./gradlew test", "./gradlew", None),
            ("cd ~/src && ./check.sh", "./check.sh", None),
            ("cd - && ./check.sh", "./check.sh", None),
            ("cd && ./check.sh", "./check.sh", None),
            ("~/bin/check", "~/bin/check", None),
        ];
        for (command, path, expected) in cases {
            assert_eq!(
                Command::normalize(command, "/work").locate(path).as_deref(),
                expected,
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
    fn relativizes_only_whole_paths() {
        let cases = [
            ("cat /Users/ada/proj/notes.txt", "cat ./notes.txt"),
            ("ls /Users/ada/proj", "ls ."),
            ("ls /Users/ada/proj/", "ls ./"),
            ("/Users/ada/proj/bin/check", "./bin/check"),
            ("cat \"/Users/ada/proj/a b.txt\"", "cat \"./a b.txt\""),
            ("cat '/Users/ada/proj'", "cat '.'"),
            (
                "node build.js --out=/Users/ada/proj/dist",
                "node build.js --out=./dist",
            ),
            (
                "docker run -v /Users/ada/proj:/app node",
                "docker run -v .:/app node",
            ),
            ("npm test > /Users/ada/proj/out.log", "npm test > ./out.log"),
            ("npm test 2>/Users/ada/proj/err.log", "npm test 2>./err.log"),
            ("(cd /Users/ada/proj/web; npm test)", "(cd ./web; npm test)"),
            ("cd /Users/ada/proj && cargo test", "cargo test"),
            ("cd /Users/ada/proj; cargo test", "cargo test"),
            ("diff /Users/ada/proj/a /Users/ada/proj/b", "diff ./a ./b"),
            (
                "cat /Users/ada/proj-old/notes.txt",
                "cat /Users/ada/proj-old/notes.txt",
            ),
            ("ls /Users/ada/project", "ls /Users/ada/project"),
            ("/Users/ada/proj2/b", "/Users/ada/proj2/b"),
            ("/mnt/Users/ada/proj/a", "/mnt/Users/ada/proj/a"),
            ("cp /Users/ada/proj.bak/a .", "cp /Users/ada/proj.bak/a ."),
            (
                "scp host:/Users/ada/proj/a .",
                "scp host:/Users/ada/proj/a .",
            ),
            (
                "cd /Users/ada/proj-old && cargo test",
                "cd /Users/ada/proj-old && cargo test",
            ),
        ];
        for cwd in ["/Users/ada/proj", "/Users/ada/proj/"] {
            for (command, expected) in cases {
                assert_eq!(
                    Command::normalize(command, cwd).text(),
                    expected,
                    "{command}"
                );
            }
        }
    }

    #[test]
    fn finds_arguments_after_relativizing() {
        let cases: [(&str, &[&str]); 3] = [
            ("cd /Users/ada/proj && npx jest --ci", &["jest", "--ci"]),
            (
                "/Users/ada/proj/bin/check --fast",
                &["./bin/check", "--fast"],
            ),
            (
                "/Users/ada/proj2/bin/check",
                &["/Users/ada/proj2/bin/check"],
            ),
        ];
        for (command, argv) in cases {
            assert_eq!(
                Command::normalize(command, "/Users/ada/proj").argv(),
                argv,
                "{command}"
            );
        }
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
            Command::normalize("cd src && cat lib.rs", "/w").action(),
            ToolAction::Read
        );
        assert_eq!(
            Command::normalize("npm test", "/w").action(),
            ToolAction::Run
        );
    }
}
