use std::sync::LazyLock;

use regex::Regex;
use trodden_redact::Redactor;

const MAX_CHARS: usize = 120;

static KINDED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:error(?:\[E\d+\])?|[A-Za-z_.]*(?:Error|Exception)|panic|fatal):\s*\S")
        .expect("error kind pattern is valid")
});

static PLAIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)command not found|no such file or directory|cannot find module|not recognized as|permission denied|module not found|unresolved import|undefined reference")
        .expect("plain error pattern is valid")
});

static ASSERTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)assert|expected|to (?:strictly )?(?:deep-?)?equal|mismatch|test(?:s)? failed|failures?=")
        .expect("assertion pattern is valid")
});

static PASSWORD_VALUE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\bpass(?:word|wd|phrase)\b\W{1,3}(?:(?:is|was|for|of)\W{1,3})?[^\s'"`]*[0-9!@#$%^&*][^\s'"`]*"#)
        .expect("password value pattern is valid")
});

pub(crate) static ANSI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-9;?]*[A-Za-z]").expect("escape pattern is valid"));

static VOLATILE: LazyLock<[(Regex, &str); 4]> = LazyLock::new(|| {
    [
        (r"(?:~|\.{0,2})/[^\s:'\x22`,)]+", "<path>"),
        (r"\b0x[0-9a-fA-F]+\b", "<hex>"),
        (r"\b\d+\b", "N"),
        (r"\s+", " "),
    ]
    .map(|(pattern, replacement)| {
        (
            Regex::new(pattern).expect("volatile pattern is valid"),
            replacement,
        )
    })
});

#[derive(Debug)]
pub struct ErrorSignature;

impl ErrorSignature {
    pub fn of(output: &str, redactor: &Redactor) -> Option<String> {
        let signature = Self::select(&redactor.redact(output))?;
        let leaks = signature.contains("[redacted:")
            || PASSWORD_VALUE.is_match(&signature)
            || redactor.contains_secret(&signature);
        (!leaks).then_some(signature)
    }

    fn select(output: &str) -> Option<String> {
        let output = ANSI.replace_all(output, "");
        let lines: Vec<&str> = output
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with("Exit code "))
            .collect();
        let traceback = lines.iter().any(|line| line.starts_with("Traceback"));
        let kinded = |line: &&&str| KINDED.is_match(line) && !ASSERTION.is_match(line);
        let line = if traceback {
            lines.iter().rev().find(kinded)
        } else {
            lines.iter().find(kinded)
        }
        .or_else(|| {
            lines
                .iter()
                .find(|line| PLAIN.is_match(line) && !ASSERTION.is_match(line))
        })?;
        Some(Self::normalize(line))
    }

    fn normalize(line: &str) -> String {
        let mut text = line.to_lowercase();
        for (pattern, replacement) in VOLATILE.iter() {
            text = pattern.replace_all(&text, *replacement).into_owned();
        }
        let text = text.trim();
        if text.chars().count() <= MAX_CHARS {
            return text.to_owned();
        }
        text.chars()
            .take(MAX_CHARS)
            .collect::<String>()
            .trim_end()
            .to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_kind_and_names_and_drops_what_varies() {
        let cases = [
            (
                "Traceback (most recent call last):\n  File \"/home/dev/app/cli.py\", line 3, in <module>\n    import yaml\nModuleNotFoundError: No module named 'yaml'",
                "modulenotfounderror: no module named 'yaml'",
            ),
            (
                "error[E0432]: unresolved import `crate::commands::total`\n --> src/cli.rs:4:5",
                "error[e0432]: unresolved import `crate::commands::total`",
            ),
            (
                "/bin/sh: line 1: python: command not found",
                "<path>: line N: python: command not found",
            ),
            (
                "Error: Cannot find module '/tmp/run-3/catalog/scripts/build.js'",
                "error: cannot find module '<path>'",
            ),
            (
                "\x1b[31mTypeError: Cannot read properties of undefined (reading 'id')\x1b[0m",
                "typeerror: cannot read properties of undefined (reading 'id')",
            ),
        ];
        for (output, signature) in cases {
            assert_eq!(
                ErrorSignature::of(output, &redactor()).as_deref(),
                Some(signature),
                "{output}"
            );
        }
    }

    #[test]
    fn ignores_test_assertions_and_unrecognized_output() {
        for output in [
            "AssertionError [ERR_ASSERTION]: Expected values to be strictly deep-equal:",
            "FAILED (failures=1)",
            "Exit code 1\n3 passing, 1 failing",
        ] {
            assert_eq!(ErrorSignature::of(output, &redactor()), None, "{output}");
        }
    }

    fn redactor() -> Redactor {
        Redactor::with_home("/Users/ada")
    }

    #[test]
    fn secrets_never_reach_a_signature() {
        let long_key = format!(
            "Error: request to the billing service was rejected for workspace eu-central-billing-prod: {}",
            ["sk", "ant", "api03", "Zq8Lm3KpQ7vX2nB9wR4tY6uI1oP5aS0dF"].join("-")
        );
        for output in [
            "Error: invalid access key AKIAIOSFODNN7EXAMPLE for bucket assets",
            "fatal: unable to access 'https://ada:hunter2@git.example.com/repo.git/': The requested URL returned error: 403",
            "fatal: unable to access 'https://ada:hunter2@localhost/repo.git/'",
            "Error: login failed for ada with password hunter2",
            "Error: API_KEY=4f9a1c2e8b7d6a5f3e2c1b0a9d8e7f6c is not valid",
            long_key.as_str(),
        ] {
            assert!(ErrorSignature::select(output).is_some(), "{output}");
            let signature = ErrorSignature::of(output, &redactor());
            assert_eq!(signature, None, "{output} gave {signature:?}");
        }
    }

    #[test]
    fn errors_that_only_mention_passwords_keep_their_signature() {
        assert_eq!(
            ErrorSignature::of(
                "Error: password authentication failed for user \"app\"",
                &redactor()
            )
            .as_deref(),
            Some("error: password authentication failed for user \"app\"")
        );
    }
}
