use std::{mem, sync::LazyLock};

use regex::Regex;
use trodden_capture::Command;

const VERIFY_PROGRAMS: &[&str] = &[
    "cargo test",
    "cargo check",
    "cargo clippy",
    "cargo build",
    "npm test",
    "pnpm test",
    "yarn test",
    "bun test",
    "deno test",
    "go test",
    "go vet",
    "go build",
    "python -m pytest",
    "python -m unittest",
    "python -m mypy",
    "pytest",
    "tox",
    "nox",
    "mypy",
    "ruff",
    "eslint",
    "tsc",
    "jest",
    "vitest",
    "rspec",
    "mix test",
    "dotnet test",
    "dotnet build",
    "mvn test",
    "mvn verify",
    "mvn package",
    "mvn install",
    "gradle test",
    "gradle check",
    "gradle build",
    "swift test",
    "phpunit",
];

const NO_RUN_FLAGS: &[&str] = &[
    "-h",
    "--help",
    "-V",
    "--version",
    "--no-run",
    "--collect-only",
    "--co",
    "--list",
    "-list",
    "--listTests",
];

static CHECK_WORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[\s/_.-])(?:test|tests|check|lint|verify|typecheck|ci)(?:$|[\s/_.:-])")
        .expect("check word pattern is valid")
});

static TEST_WORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[\s/_.:-])(?:tests?|specs?|unittest|pytest|jest|vitest|mocha|rspec|phpunit|nextest|tox|nox)(?:$|[\s/_.:-])")
        .expect("test word pattern is valid")
});

static TRAILING_FILTER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\s*(?:2>&1|\|&?\s*(?:head|tail|grep|less|more|tee|cat|wc)\b.*|\|\|\s*(?:true|:)|;\s*echo\b.*)$")
        .expect("filter pattern is valid")
});

#[derive(Debug)]
pub(crate) struct Verification;

impl Verification {
    pub(crate) fn is_verify(command: &str) -> bool {
        let command = Command::normalize(&Self::clean(command), "");
        let argv = command.argv();
        let Some((executable, arguments)) = argv.split_first() else {
            return false;
        };
        if arguments.iter().any(|argument| {
            NO_RUN_FLAGS.contains(
                &argument
                    .split_once('=')
                    .map_or(argument.as_str(), |(flag, _)| flag),
            )
        }) {
            return false;
        }
        let program = command.program();
        if VERIFY_PROGRAMS.contains(&program.as_str()) {
            return true;
        }
        let targets = arguments.join(" ");
        let runner = program.split_whitespace().next().unwrap_or_default();
        match program.as_str() {
            "node" => arguments.iter().any(|argument| argument == "--test"),
            "cargo nextest" => arguments
                .iter()
                .any(|argument| argument == "run" || argument == "r"),
            "npm run" | "pnpm run" | "yarn run" | "bun run" => CHECK_WORD.is_match(&targets),
            _ if ["make", "just", "rake", "sh", "bash", "zsh"].contains(&runner) => {
                CHECK_WORD.is_match(&targets)
            }
            _ if ["mvn", "gradle"].contains(&runner) => command
                .operands()
                .iter()
                .any(|task| Self::is_check_task(runner, task)),
            _ if program.starts_with("python ") => {
                Self::is_check_script(&program, arguments.get(1))
            }
            _ if executable.contains('/') => Self::is_check_script(executable, arguments.first()),
            _ => false,
        }
    }

    fn is_check_task(runner: &str, task: &str) -> bool {
        let task = task.rsplit(':').next().unwrap_or(task);
        VERIFY_PROGRAMS.contains(&format!("{runner} {task}").as_str())
    }

    fn is_check_script(script: &str, task: Option<&String>) -> bool {
        CHECK_WORD.is_match(script)
            || task.is_some_and(|task| {
                !task.starts_with('-') && !task.contains(['/', '.']) && CHECK_WORD.is_match(task)
            })
    }

    pub(crate) fn runs_tests(command: &str) -> bool {
        TEST_WORD.is_match(&Self::clean(command))
    }

    pub(crate) fn covers(passed: &str, failed: &str) -> bool {
        if Self::runs_tests(failed) {
            return Self::runs_tests(passed);
        }
        let program = |command: &str| Command::normalize(&Self::clean(command), "").program();
        program(passed) == program(failed)
    }

    pub(crate) fn hides_status(command: &str) -> bool {
        let segments = Self::segments(command);
        let Some(check) = segments.iter().position(|(segment, separator)| {
            segment.split_whitespace().next() != Some("cd")
                || !["&&", ";", "\n"].contains(separator)
        }) else {
            return false;
        };
        segments[check..]
            .windows(2)
            .any(|pair| pair[0].1 != "&&" && !pair[1].0.trim().is_empty())
    }

    fn segments(command: &str) -> Vec<(String, &'static str)> {
        let chars: Vec<char> = command.chars().collect();
        let mut segments = Vec::new();
        let mut segment = String::new();
        let mut quote = None;
        let mut index = 0;
        while let Some(&c) = chars.get(index) {
            let next = chars.get(index + 1).copied();
            let separator = match (quote, c) {
                (Some(open), _) if c == open => {
                    quote = None;
                    None
                }
                (Some('\''), _) => None,
                (_, '\\') => {
                    segment.push(c);
                    segment.extend(next);
                    index += 2;
                    continue;
                }
                (Some(_), _) => None,
                (None, '\'' | '"') => {
                    quote = Some(c);
                    None
                }
                (None, '|') => Some(match next {
                    Some('|') => "||",
                    Some('&') => "|&",
                    _ => "|",
                }),
                (None, '&') if next == Some('&') => Some("&&"),
                (None, '&')
                    if index.checked_sub(1).map(|before| chars[before]) == Some('>')
                        || next == Some('>') =>
                {
                    None
                }
                (None, '&') => Some("&"),
                (None, ';') => Some(";"),
                (None, '\n') => Some("\n"),
                (None, _) => None,
            };
            if let Some(separator) = separator {
                segments.push((mem::take(&mut segment), separator));
                index += separator.len();
            } else {
                segment.push(c);
                index += 1;
            }
        }
        segments.push((segment, ""));
        segments
    }

    pub(crate) fn clean(command: &str) -> String {
        let mut text = command.trim().to_owned();
        loop {
            let trimmed = TRAILING_FILTER.replace(&text, "").trim_end().to_owned();
            if trimmed == text {
                return text;
            }
            text = trimmed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_check_commands() {
        for command in [
            "cargo test -p invoicer",
            "npm test",
            "python3 -m unittest discover -s tests",
            "node --test \"test/**/*.test.js\"",
            "npm run lint",
            "make check",
            "python3 tools/check.py",
            "sh scripts/test.sh",
            "cargo test --lib 2>&1 | head -50",
            "./bin/check",
            "bin/check",
            "./scripts/test.sh",
            "./gradlew test",
            "./gradlew check",
            "./mvnw -q verify",
            "cd crates/x && cargo test",
            "cd web && npm test",
            "cd web && npm run lint",
            "npx jest",
            "npx tsc --noEmit",
            "uv run pytest",
            "uv run python -m pytest",
            "poetry run pytest",
            "bundle exec rspec",
            "bundle exec rake test",
            "python3 manage.py test",
            "./x.py test",
            "cargo +nightly test",
            "cargo nextest run",
            "time cargo test",
            "timeout 600 cargo test",
            "FOO=\"a b\" cargo test",
            "sudo -u ci make check",
            "make -C web check",
        ] {
            assert!(Verification::is_verify(command), "{command}");
        }
        for command in [
            "python3 tools/gen_models.py",
            "python3 tools/gen_models.py --check",
            "python3 tools/seed.py tests/fixtures.json",
            "npm run gen",
            "cargo run -- count",
            "git status",
            "make install",
            "./scripts/gen.sh",
            "cd test && npm run build",
            "npx prettier --write .",
            "uv run alembic upgrade head",
        ] {
            assert!(!Verification::is_verify(command), "{command}");
        }
    }

    #[test]
    fn ignores_check_programs_that_run_no_checks() {
        for command in [
            "ruff --version",
            "ruff -V",
            "cargo test --help",
            "cargo test -h",
            "cargo test --no-run",
            "cargo test -- --list",
            "cargo nextest list",
            "pytest --collect-only",
            "pytest --co -q",
            "go test -list .",
            "go test -list=Paging ./...",
            "npx jest --listTests",
            "make check --help",
            "./bin/check --help",
            "cd web && npm test -- --help",
        ] {
            assert!(!Verification::is_verify(command), "{command}");
        }
    }

    #[test]
    fn recognizes_runners_given_several_tasks() {
        let cases = [
            ("mvn clean test", true),
            ("mvn clean verify", true),
            ("./mvnw -q clean install verify", true),
            ("mvn -pl core clean surefire:test", true),
            ("mvn clean verify -DskipTests", true),
            ("mvn clean install", true),
            ("./mvnw clean package", true),
            ("./gradlew clean build", true),
            ("./gradlew clean check -x test", true),
            ("cd android && ./gradlew clean :app:test", true),
            ("gradle --offline clean test --tests InvoiceTest", true),
            ("make clean test", true),
            ("just fmt lint", true),
            ("mvn dependency:tree", false),
            ("./gradlew clean assemble", false),
            ("gradle help --task test", false),
            ("mvn clean test --help", false),
            ("./gradlew clean build --version", false),
        ];
        for (command, verifies) in cases {
            assert_eq!(Verification::is_verify(command), verifies, "{command}");
        }
    }

    #[test]
    fn tells_test_runs_from_other_checks() {
        for command in [
            "cargo test -p invoicer",
            "npm test",
            "node --test test/",
            "python3 -m pytest -q",
            "npm run test:unit",
        ] {
            assert!(Verification::runs_tests(command), "{command}");
        }
        for command in [
            "cargo build",
            "ruff check .",
            "npm run lint",
            "tsc --noEmit",
        ] {
            assert!(!Verification::runs_tests(command), "{command}");
        }
    }

    #[test]
    fn finds_what_hides_a_checks_exit_status() {
        for command in [
            "cargo test 2>&1 | tail -20",
            "cargo test |& tee test.log",
            "cargo test | sort",
            "npm test || true",
            "npm test || exit 0",
            "npm test; echo ok",
            "npm test & wait",
            "cd web && npm test | head",
            "cargo test && echo ok; true",
            "pytest -q\necho done",
        ] {
            assert!(Verification::hides_status(command), "{command}");
        }
        for command in [
            "cargo test",
            "cargo test 2>&1",
            "npm test &> out.log",
            "npm test >&2",
            "cargo build && cargo test",
            "cargo test && echo ok",
            "cd web && npm test",
            "cd web; npm test",
            "cd web\nnpm test",
            "npm test;",
            "pytest -k 'paging | totals'",
            "pytest -k \"paging; totals\"",
            "grep -q a\\|b x && pytest",
        ] {
            assert!(!Verification::hides_status(command), "{command}");
        }
    }

    #[test]
    fn strips_output_filters() {
        assert_eq!(
            Verification::clean("cargo test --lib 2>&1 | head -50"),
            "cargo test --lib"
        );
        assert_eq!(Verification::clean("npm test | tee out.log"), "npm test");
        assert_eq!(Verification::clean("pytest -q"), "pytest -q");
        assert_eq!(Verification::clean("cargo test || true"), "cargo test");
        assert_eq!(Verification::clean("npm test; echo ok"), "npm test");
    }
}
