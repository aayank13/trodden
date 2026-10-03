use std::sync::LazyLock;

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
            _ if program.starts_with("python ") => {
                Self::is_check_script(&program, arguments.get(1))
            }
            _ if executable.contains('/') => Self::is_check_script(executable, arguments.first()),
            _ => false,
        }
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
