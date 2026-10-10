use std::sync::LazyLock;

use regex::Regex;

use crate::{command::Command, error::ANSI};

static FAILED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^\s*(?:test result: FAILED|error(?:\[E\d+\])?:|FAILED\b|FAIL\b|--- FAIL|npm (?:ERR!|error)|# fail [1-9]|ℹ fail [1-9])|(?:^|[\s(,])[1-9]\d* (?:failed|failing)(?:[\s,;.)]|$)|\berror TS\d+:|\([1-9]\d* errors?,",
    )
    .expect("failure pattern is valid")
});

static PASSED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^\s*(?:test result: ok\.|# pass [1-9]|ℹ pass [1-9]|ok\s+\S+\s+[\d.]+s|All checks passed!|Finished\b.*\btarget\(s\))|(?:^|[\s(,])[1-9]\d* passed(?:[\s,;.)]|$)",
    )
    .expect("success pattern is valid")
});

static CHECK_WORD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(?:^|[/_.-])(?:test|tests|check|lint|verify|typecheck|ci)(?:$|[/_.-])")
        .expect("check word pattern is valid")
});

#[derive(Debug)]
pub(crate) struct CheckOutput;

impl CheckOutput {
    const CHECK_PROGRAMS: &[&str] = &[
        "cargo test",
        "cargo check",
        "cargo clippy",
        "cargo build",
        "cargo nextest",
        "npm test",
        "pnpm test",
        "yarn test",
        "bun test",
        "deno test",
        "npm run",
        "pnpm run",
        "yarn run",
        "bun run",
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
        "mocha",
        "rspec",
        "mix test",
        "dotnet test",
        "dotnet build",
        "swift test",
        "phpunit",
    ];

    const TASK_RUNNERS: &[&str] = &["make", "just", "rake", "mvn", "gradle"];

    const SHELLS: &[&str] = &["sh", "bash", "zsh"];

    pub(crate) fn verdict(command: &str, output: &str) -> Option<bool> {
        if !Self::hides_exit_code(command) || !Self::runs_check(command) {
            return None;
        }
        let output = ANSI.replace_all(output, "");
        if FAILED.is_match(&output) {
            Some(false)
        } else {
            PASSED.is_match(&output).then_some(true)
        }
    }

    fn runs_check(command: &str) -> bool {
        let command = Command::normalize(command, "");
        let program = command.program();
        let runner = program.split_whitespace().next().unwrap_or_default();
        let argv = command.argv();
        let script = if Self::SHELLS.contains(&runner) {
            argv.get(1)
        } else {
            argv.first().filter(|executable| executable.contains('/'))
        };
        Self::CHECK_PROGRAMS.contains(&program.as_str())
            || Self::TASK_RUNNERS.contains(&runner)
            || script.is_some_and(|script| CHECK_WORD.is_match(script))
    }

    fn hides_exit_code(command: &str) -> bool {
        let chars: Vec<char> = command.chars().collect();
        let mut quote = None;
        let mut index = 0;
        while index < chars.len() {
            let c = chars[index];
            match quote {
                Some(open) if c == open => quote = None,
                Some('"') if c == '\\' => index += 1,
                Some(_) => {}
                None => match c {
                    '\'' | '"' => quote = Some(c),
                    '\\' => index += 1,
                    ';' | '\n' | '|' => return true,
                    '&' if chars.get(index + 1) == Some(&'&') => index += 1,
                    '&' if index.checked_sub(1).map(|before| chars[before]) != Some('>')
                        && chars.get(index + 1) != Some(&'>') =>
                    {
                        return true;
                    }
                    _ => {}
                },
            }
            index += 1;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_compound_commands_are_judged_by_output() {
        let failing = "test result: FAILED. 2 passed; 1 failed";
        for command in [
            "cargo test 2>&1 | tail -20",
            "cargo test |& tee test.log",
            "cargo test || true",
            "npm test; echo ok",
            "npm test & wait",
        ] {
            assert_eq!(
                CheckOutput::verdict(command, failing),
                Some(false),
                "{command}"
            );
        }
        for command in [
            "cargo test",
            "cargo test 2>&1",
            "npm test &> out.log",
            "cargo build && cargo test",
            "pytest -k 'paging | totals'",
        ] {
            assert_eq!(CheckOutput::verdict(command, failing), None, "{command}");
        }
    }

    #[test]
    fn only_checks_are_judged_by_output() {
        let printed = "fn lookup(id: u32) {\nerror: unknown customer id\n}\ndone";
        for command in [
            "sed -n '1,40p' src/report.rs; echo done",
            "cat src/report.rs | head -40",
            "git log --oneline | head",
            "echo 'test result: FAILED'; true",
            "python scripts/gen.py | tail",
            "bash -c 'cat notes.txt' | head",
        ] {
            assert_eq!(CheckOutput::verdict(command, printed), None, "{command}");
        }
        for command in [
            "cd api && cargo test 2>&1 | tail -20",
            "npx jest --ci | tail",
            "uv run pytest -q || true",
            "make check | tail",
            "./scripts/test.sh 2>&1 | tail",
            "bash ci.sh; echo exit=$?",
        ] {
            assert_eq!(
                CheckOutput::verdict(command, printed),
                Some(false),
                "{command}"
            );
        }
    }

    #[test]
    fn reads_common_runner_summaries() {
        let piped = "cargo test | tail";
        for output in [
            "test result: FAILED. 0 passed; 1 failed",
            "error[E0308]: mismatched types\n --> src/lib.rs:4:5",
            "error: could not compile `invoicer`",
            "=== 1 failed, 4 passed in 0.31s ===",
            "  2 passing (12ms)\n  1 failing",
            "# fail 1",
            "\u{1b}[31mℹ fail 2\u{1b}[0m",
            "--- FAIL: TestPaging (0.00s)",
            "npm ERR! Test failed.",
            "src/app.ts(3,5): error TS2322: Type 'string' is not assignable",
            "✖ 3 problems (2 errors, 1 warning)",
        ] {
            assert_eq!(CheckOutput::verdict(piped, output), Some(false), "{output}");
        }
        for output in [
            "test result: ok. 3 passed; 0 failed; 0 ignored",
            "=== 5 passed in 0.12s ===",
            "# pass 3\n# fail 0",
            "ok  \texample.com/paging\t0.012s",
            "All checks passed!",
            "    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.31s",
        ] {
            assert_eq!(CheckOutput::verdict(piped, output), Some(true), "{output}");
        }
        assert_eq!(
            CheckOutput::verdict(piped, "line 18\nline 19\nline 20"),
            None
        );
        assert_eq!(
            CheckOutput::verdict(piped, "TOTAL passed=180 failed=0"),
            None
        );
        assert_eq!(
            CheckOutput::verdict(
                piped,
                "TOTAL passed=180 failed=0\ntest result: ok. 180 passed"
            ),
            Some(true)
        );
    }
}
