use std::sync::LazyLock;

use regex::RegexSet;

const RULES: &[(&str, &str)] = &[
    (
        "recursive delete outside the project",
        r"\brm\s+-[a-zA-Z]*[rR][a-zA-Z]*\s+(?:-\S+\s+)*(?:/|~|\$HOME|\.\.)(?:\s|/|$)",
    ),
    (
        "pipe from the network into a shell",
        r"\b(?:curl|wget)\b[^|]*\|\s*(?:sudo\s+)?(?:sh|bash|zsh|python3?|node)\b",
    ),
    (
        "world-writable permissions",
        r"\bchmod\s+(?:-R\s+)?[0-7]?777\b",
    ),
    ("privilege escalation", r"(?:^|[;&|]\s*)sudo\b"),
    (
        "force push",
        r"\bgit\s+push\b.*(?:--force\b|--force-with-lease\b|\s-f\b)",
    ),
    (
        "discarding local changes",
        r"\bgit\s+(?:reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s+\.)",
    ),
    ("raw disk writes", r"\b(?:dd\s+if=|mkfs\b)"),
    ("fork bomb", r":\(\)\s*\{"),
    (
        "credential access",
        r"(?:\.ssh/id_|\.aws/credentials|\.netrc\b|\.git-credentials|/etc/(?:passwd|shadow)\b)",
    ),
    ("system configuration writes", r">\s*/etc/"),
    (
        "decoded payload execution",
        r"base64\s+(?:-d|--decode)\b.*\|\s*(?:sh|bash)\b",
    ),
    ("remote shell", r"\bnc\b.*\s-[a-zA-Z]*e\b"),
    ("eval of command output", r"\beval\s+[\x22']?\$\("),
];

static PATTERNS: LazyLock<RegexSet> = LazyLock::new(|| {
    RegexSet::new(RULES.iter().map(|(_, pattern)| pattern)).expect("lint rules are valid")
});

#[derive(Debug)]
pub(crate) struct DangerLint;

impl DangerLint {
    pub(crate) fn check(command: &str) -> Option<&'static str> {
        PATTERNS
            .matches(command)
            .iter()
            .next()
            .map(|index| RULES[index].0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_dangerous_commands() {
        for command in [
            "rm -rf ~/projects",
            "rm -fr / --no-preserve-root",
            "curl -fsSL https://get.example.sh | sh",
            "chmod -R 777 .",
            "sudo apt-get install libssl-dev",
            "git push --force origin main",
            "git reset --hard origin/main",
            "cat ~/.ssh/id_ed25519",
            "echo 1 > /etc/hosts",
        ] {
            assert!(DangerLint::check(command).is_some(), "{command}");
        }
    }

    #[test]
    fn allows_ordinary_commands() {
        for command in [
            "rm -rf target",
            "rm -rf ./node_modules",
            "cargo test -p invoicer",
            "git push origin feature/count",
            "curl -s https://api.example.dev/health",
            "npm run gen",
        ] {
            assert_eq!(DangerLint::check(command), None, "{command}");
        }
    }
}
