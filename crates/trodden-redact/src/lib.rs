use std::borrow::Cow;

use regex::{Captures, Regex};

const MARKER_PREFIX: &str = "[REDACTED:";

const ENTROPY_MIN_LEN: usize = 32;

const ENTROPY_THRESHOLD: f64 = 4.3;

#[derive(Debug, Clone)]
pub struct Redactor {
    rules: Vec<Rule>,
    token: Regex,
    home: Option<String>,
}

#[derive(Debug, Clone)]
struct Rule {
    id: &'static str,
    regex: Regex,
}

impl Rule {
    const SOURCES: &[(&'static str, &'static str)] = &[
        (
            "private-key",
            r"(?s)(?P<secret>-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----)",
        ),
        (
            "aws-access-key",
            r"\b(?P<secret>(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16})\b",
        ),
        (
            "github-token",
            r"\b(?P<secret>gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})\b",
        ),
        ("gitlab-token", r"\b(?P<secret>glpat-[A-Za-z0-9_-]{20,})"),
        ("slack-token", r"\b(?P<secret>xox[abprs]-[A-Za-z0-9-]{10,})"),
        (
            "slack-webhook",
            r"(?P<secret>https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+)",
        ),
        (
            "stripe-key",
            r"\b(?P<secret>[rs]k_(?:live|test)_[A-Za-z0-9]{16,})\b",
        ),
        ("google-api-key", r"\b(?P<secret>AIza[0-9A-Za-z_-]{35})"),
        (
            "llm-api-key",
            r"\b(?P<secret>sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,})",
        ),
        ("npm-token", r"\b(?P<secret>npm_[A-Za-z0-9]{36})\b"),
        (
            "jwt",
            r"\b(?P<secret>eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})",
        ),
        (
            "url-password",
            r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<secret>[^/\s@]+)@",
        ),
        (
            "auth-header",
            r"(?i)\bauthorization:\s*(?:bearer|basic|token)\s+(?P<secret>[A-Za-z0-9._~+/=-]{8,})",
        ),
        (
            "assignment",
            r#"(?i)\b[A-Z0-9_]*(?:SECRET|TOKEN|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIALS?)[A-Z0-9_]*\s*[=:]\s*(?P<secret>"[^"]*"|'[^']*'|[^\s"']+)"#,
        ),
        (
            "flag",
            r"(?i)--(?:password|passwd|token|secret|api-key|apikey|access-key|auth)[= ](?P<secret>\S+)",
        ),
        (
            "email",
            r"\b(?P<secret>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b",
        ),
    ];
}

impl Redactor {
    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok();
        Self::build(home)
    }

    pub fn with_home(home: impl Into<String>) -> Self {
        Self::build(Some(home.into()))
    }

    fn build(home: Option<String>) -> Self {
        let rules = Rule::SOURCES
            .iter()
            .map(|(id, source)| Rule {
                id,
                regex: Regex::new(source).expect("built-in rules are valid regexes"),
            })
            .collect();
        let token = Regex::new(r"[A-Za-z0-9+/=_-]{32,}").expect("token pattern is valid");
        let home = home
            .filter(|home| home.len() > 1)
            .map(|home| home.trim_end_matches('/').to_owned());
        Self { rules, token, home }
    }

    #[must_use]
    pub fn redact<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut text = Cow::Borrowed(text);
        for rule in &self.rules {
            if let Cow::Owned(replaced) = rule.regex.replace_all(&text, |caps: &Captures<'_>| {
                Self::replace_secret(rule.id, caps)
            }) {
                text = Cow::Owned(replaced);
            }
        }
        if let Cow::Owned(replaced) = self
            .token
            .replace_all(&text, |caps: &Captures<'_>| Self::redact_random(&caps[0]))
        {
            text = Cow::Owned(replaced);
        }
        match &self.home {
            Some(home) if text.contains(home.as_str()) => Cow::Owned(Self::fold_home(&text, home)),
            _ => text,
        }
    }

    pub fn contains_secret(&self, text: &str) -> bool {
        self.redact_secrets_only(text) != text
    }

    fn redact_secrets_only(&self, text: &str) -> String {
        let without_home = Self {
            home: None,
            ..self.clone()
        };
        without_home.redact(text).into_owned()
    }

    fn replace_secret(rule: &str, caps: &Captures<'_>) -> String {
        let whole = caps.get(0).expect("group 0 always participates");
        let Some(secret) = caps.name("secret") else {
            return format!("{MARKER_PREFIX}{rule}]");
        };
        if secret.as_str().starts_with(MARKER_PREFIX) {
            return whole.as_str().to_owned();
        }
        let start = secret.start() - whole.start();
        let end = secret.end() - whole.start();
        let original = whole.as_str();
        format!(
            "{}{MARKER_PREFIX}{rule}]{}",
            &original[..start],
            &original[end..]
        )
    }

    fn fold_home(text: &str, home: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(index) = rest.find(home) {
            let after = &rest[index + home.len()..];
            let at_boundary =
                after.is_empty() || after.starts_with(['/', '\\', ' ', '"', '\'', ':']);
            out.push_str(&rest[..index]);
            out.push_str(if at_boundary { "~" } else { home });
            rest = after;
        }
        out.push_str(rest);
        out
    }

    fn redact_random(token: &str) -> String {
        let marker = format!("{MARKER_PREFIX}high-entropy]");
        if !token.starts_with(['/', '~', '.']) {
            return if Self::looks_random(token) {
                marker
            } else {
                token.to_owned()
            };
        }
        token
            .split('/')
            .map(|segment| {
                if Self::looks_random(segment) {
                    marker.as_str()
                } else {
                    segment
                }
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    fn looks_random(token: &str) -> bool {
        if token.len() < ENTROPY_MIN_LEN
            || !token.bytes().any(|b| b.is_ascii_digit())
            || !token.bytes().any(|b| b.is_ascii_alphabetic())
        {
            return false;
        }
        let mut counts = [0_u32; 256];
        for byte in token.bytes() {
            counts[usize::from(byte)] += 1;
        }
        let len = token.len() as f64;
        let entropy: f64 = counts
            .iter()
            .filter(|&&count| count > 0)
            .map(|&count| {
                let p = f64::from(count) / len;
                -p * p.log2()
            })
            .sum();
        entropy > ENTROPY_THRESHOLD
    }
}

impl Default for Redactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;

    use super::*;

    #[test]
    fn redacts_secrets_and_folds_home() {
        let redactor = Redactor::with_home("/Users/ada");
        let key = ["sk", "live", "4eC39HqLyjWDarjtT1zdp7dc"].join("_");
        let command = format!("STRIPE_KEY={key} node /Users/ada/shop/seed.js");

        assert_eq!(
            redactor.redact(&command),
            "STRIPE_KEY=[REDACTED:stripe-key] node ~/shop/seed.js",
        );
    }

    #[test]
    fn keeps_paths_with_random_looking_segments() {
        let redactor = Redactor::with_home("/Users/ada");
        let path = "/private/var/folders/q3/x8k2mz7pd4vb19tn6wr5c0hs0000gn/T/scratch/run-3/catalog";

        assert_eq!(redactor.redact(path), path);
    }

    #[test]
    fn detects_remaining_secrets() {
        let redactor = Redactor::with_home("/Users/ada");

        assert!(redactor.contains_secret("SECRET_TOKEN=abc123"));
        assert!(!redactor.contains_secret("cargo test -p invoicer"));
        assert!(!redactor.contains_secret("cat /Users/ada/notes.md"));
    }

    quickcheck! {
        fn redaction_is_idempotent(text: String) -> bool {
            let redactor = Redactor::with_home("/Users/ada");
            let once = redactor.redact(&text).into_owned();
            redactor.redact(&once) == once
        }
    }
}
