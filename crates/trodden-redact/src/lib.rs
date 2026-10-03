use std::borrow::Cow;

use regex::{Captures, Regex};

const MARKER_PREFIX: &str = "[REDACTED:";

const ENTROPY_MIN_LEN: usize = 32;

const ENTROPY_THRESHOLD: f64 = 4.3;

const PATH_WORD_MIN_LEN: usize = 3;

type Check = fn(&Captures<'_>) -> bool;

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
    is_harmless: Check,
}

impl Rule {
    const SOURCES: &[(&'static str, &'static str, Check)] = &[
        (
            "private-key",
            r"(?s)(?P<secret>-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----.*?-----END [A-Z ]*PRIVATE KEY(?: BLOCK)?-----)",
            Self::never,
        ),
        (
            "aws-access-key",
            r"(?-u:\b)(?P<secret>(?:AKIA|ASIA|ABIA|ACCA)[A-Z0-9]{16})(?-u:\b)",
            Self::never,
        ),
        (
            "github-token",
            r"(?-u:\b)(?P<secret>gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{50,})(?-u:\b)",
            Self::never,
        ),
        (
            "gitlab-token",
            r"(?-u:\b)(?P<secret>glpat-[A-Za-z0-9_-]{20,})",
            Self::never,
        ),
        (
            "slack-token",
            r"(?-u:\b)(?P<secret>xox[abprs]-[A-Za-z0-9-]{10,})",
            Self::never,
        ),
        (
            "slack-webhook",
            r"(?P<secret>https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+)",
            Self::never,
        ),
        (
            "stripe-key",
            r"(?-u:\b)(?P<secret>[rs]k_(?:live|test)_[A-Za-z0-9]{16,})(?-u:\b)",
            Self::never,
        ),
        (
            "google-api-key",
            r"(?-u:\b)(?P<secret>AIza[0-9A-Za-z_-]{35})",
            Self::never,
        ),
        (
            "llm-api-key",
            r"(?-u:\b)(?P<secret>sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,})",
            Self::never,
        ),
        (
            "npm-token",
            r"(?-u:\b)(?P<secret>npm_[A-Za-z0-9]{36})(?-u:\b)",
            Self::never,
        ),
        (
            "jwt",
            r"(?-u:\b)(?P<secret>eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})",
            Self::never,
        ),
        (
            "url-password",
            r"(?i)(?-u:\b)[a-z][a-z0-9+.-]*://[^/\s:@]+:(?P<secret>[^/\s@]+)@",
            Self::never,
        ),
        (
            "auth-header",
            r"(?i)(?-u:\b)authorization:\s*(?:bearer|basic|token)\s+(?P<secret>[A-Za-z0-9._~+/=-]{8,})",
            Self::never,
        ),
        (
            "cookie",
            r#"(?i)(?-u:\b)(?:set-)?cookie["']?\s*:\s*(?P<secret>[^\s"'](?:[^\r\n"']*[^\s"'])?)"#,
            Self::is_reference,
        ),
        (
            "assignment",
            r#"(?i)(?P<name>[a-z0-9_.-]*(?:secret|token|pass|api[_.-]?key|access[_.-]?key|private[_.-]?key|credential)[a-z0-9_.-]*)["']?\s*[=:]\s*(?P<secret>"[^"]*"|'[^']*'|[^\s"'=:][^\s"']*)"#,
            Self::is_harmless_assignment,
        ),
        (
            "flag",
            r#"(?i)(?:^|[\s"'])(?P<name>--?[a-z0-9_.-]*?(?:secret|token|pass|passw(?:or)?d|passphrase|key|credentials?|auth|cookie))(?:=|\s+)(?P<secret>"[^"]*"|'[^']*'|[^\s"'-][^\s"']*)"#,
            Self::is_harmless_flag,
        ),
        (
            "cli-password",
            r#"(?-u:\b)curl(?-u:\b)[^\n|;&]*?\s(?:-u|--user)(?:\s+|=)?["']?[^\s:"']*:(?P<secret>[^\s"']+)"#,
            Self::is_reference,
        ),
        (
            "cli-password",
            r#"(?-u:\b)(?i:mysql[a-z]*|mariadb[a-z-]*)(?-u:\b)[^\n|;&]*?\s-p["']?(?P<secret>[^\s"']+)"#,
            Self::is_reference,
        ),
        (
            "cli-password",
            r#"(?-u:\b)sshpass(?:\s+-v)*\s+-p\s*["']?(?P<secret>[^\s"']+)"#,
            Self::is_reference,
        ),
        (
            "hex-secret",
            r#"(?P<name>[A-Za-z0-9_.-]+)["']?\s*[=:]\s*["']?(?P<secret>[0-9A-Fa-f]{32,})(?-u:\b)"#,
            Self::is_harmless_hex,
        ),
        (
            "email",
            r"(?-u:\b)(?P<secret>(?P<user>[A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Za-z]{2,})(?-u:\b)(?P<remote>[:/])?",
            Self::is_git_remote,
        ),
    ];

    fn never(_: &Captures<'_>) -> bool {
        false
    }

    fn is_reference(caps: &Captures<'_>) -> bool {
        Value::new(&caps["secret"]).is_reference()
    }

    fn is_harmless_assignment(caps: &Captures<'_>) -> bool {
        let name = Name::parse(&caps["name"]);
        let value = Value::new(&caps["secret"]);
        value.is_single()
            && (name.is_qualified()
                || !name.mentions_secret()
                || value.is_harmless(name.ends_with_secret()))
    }

    fn is_harmless_flag(caps: &Captures<'_>) -> bool {
        let name = Name::parse(&caps["name"]);
        let value = Value::new(&caps["secret"]);
        value.is_single() && (!name.is_secret_flag() || value.is_harmless(true))
    }

    fn is_harmless_hex(caps: &Captures<'_>) -> bool {
        let hex = &caps["secret"];
        Name::parse(&caps["name"]).names_hash()
            || !hex.bytes().any(|b| b.is_ascii_digit())
            || !hex.bytes().any(|b| b.is_ascii_alphabetic())
    }

    fn is_git_remote(caps: &Captures<'_>) -> bool {
        &caps["user"] == "git" && caps.name("remote").is_some()
    }
}

#[derive(Debug)]
struct Name {
    words: Vec<String>,
    joined: String,
}

impl Name {
    const SECRET_WORDS: &[&str] = &[
        "secret",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "apikey",
        "accesskey",
        "privatekey",
    ];

    const PASS_WORDS: &[&str] = &["pass", "pgpass", "sshpass", "dbpass"];

    const QUALIFIERS: &[&str] = &["file", "files", "path", "dir", "type"];

    const HASH_WORDS: &[&str] = &[
        "sha",
        "sha1",
        "sha256",
        "sha384",
        "sha512",
        "md5",
        "hash",
        "digest",
        "checksum",
        "commit",
        "rev",
        "revision",
        "ref",
        "head",
        "base",
        "parent",
        "tree",
        "blob",
        "object",
        "oid",
        "etag",
        "integrity",
        "uuid",
        "guid",
        "version",
    ];

    fn parse(name: &str) -> Self {
        let mut words = Vec::new();
        let mut word = String::new();
        let mut previous = None;
        for c in name.chars() {
            let camel = c.is_ascii_uppercase()
                && previous.is_some_and(|p: char| p.is_ascii_lowercase() || p.is_ascii_digit());
            if (camel || !c.is_ascii_alphanumeric()) && !word.is_empty() {
                words.push(std::mem::take(&mut word));
            }
            if c.is_ascii_alphanumeric() {
                word.push(c.to_ascii_lowercase());
            }
            previous = Some(c);
        }
        if !word.is_empty() {
            words.push(word);
        }
        let joined = words
            .concat()
            .trim_end_matches(|c: char| c.is_ascii_digit())
            .to_owned();
        Self { words, joined }
    }

    fn first(&self) -> &str {
        self.words.first().map_or("", String::as_str)
    }

    fn last(&self) -> &str {
        self.words.last().map_or("", String::as_str)
    }

    fn is_qualified(&self) -> bool {
        Self::QUALIFIERS.contains(&self.last())
    }

    fn mentions_secret(&self) -> bool {
        Self::SECRET_WORDS
            .iter()
            .any(|word| self.joined.contains(word))
            || self.joined.match_indices("token").any(|(index, _)| {
                let rest = &self.joined[index + "token".len()..];
                !rest.starts_with('s') && !rest.starts_with("iz")
            })
            || self
                .words
                .iter()
                .any(|word| Self::PASS_WORDS.contains(&word.as_str()))
    }

    fn ends_with_secret(&self) -> bool {
        Self::SECRET_WORDS
            .iter()
            .chain(&["token", "credentials"])
            .any(|word| self.joined.ends_with(word))
            || Self::PASS_WORDS.contains(&self.last())
            || (self.joined.ends_with("key") && self.mentions_secret())
    }

    fn is_secret_flag(&self) -> bool {
        self.first() != "no"
            && (self.ends_with_secret() || ["auth", "cookie"].contains(&self.last()))
    }

    fn names_hash(&self) -> bool {
        self.words
            .iter()
            .any(|word| Self::HASH_WORDS.contains(&word.as_str()))
    }
}

#[derive(Debug)]
struct Value<'a>(&'a str);

impl<'a> Value<'a> {
    const BOOLEANS: &'static [&'static str] = &[
        "true", "false", "yes", "no", "on", "off", "none", "null", "nil",
    ];

    fn new(raw: &'a str) -> Self {
        let unquoted = ['"', '\'']
            .iter()
            .find_map(|&quote| raw.strip_prefix(quote)?.strip_suffix(quote))
            .unwrap_or(raw);
        let trimmed = unquoted.trim_end_matches([',', ';', ')', ']']);
        if trimmed.contains('{') {
            Self(trimmed)
        } else {
            Self(trimmed.trim_end_matches('}'))
        }
    }

    fn is_single(&self) -> bool {
        !self.0.contains(['=', '&', ',', ';'])
    }

    fn is_harmless(&self, numbers_are_secret: bool) -> bool {
        self.0.is_empty()
            || Self::BOOLEANS
                .iter()
                .any(|word| self.0.eq_ignore_ascii_case(word))
            || (!numbers_are_secret && self.is_number())
            || self.is_reference()
            || self.is_plain_path()
    }

    fn is_number(&self) -> bool {
        let digits = self.0.strip_prefix('-').unwrap_or(self.0);
        digits.bytes().any(|b| b.is_ascii_digit())
            && digits.bytes().all(|b| b.is_ascii_digit() || b == b'.')
            && digits.matches('.').count() <= 1
    }

    fn is_reference(&self) -> bool {
        let Some(name) = self.0.strip_prefix('$') else {
            return false;
        };
        let name = name
            .strip_prefix('{')
            .and_then(|name| name.strip_suffix('}'))
            .unwrap_or(name);
        name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
    }

    fn is_plain_path(&self) -> bool {
        let Some(rest) = ["/", "~/", "./", "../"]
            .iter()
            .find_map(|prefix| self.0.strip_prefix(prefix))
        else {
            return false;
        };
        rest.split('/').all(|segment| {
            segment
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
                && !Self::looks_generated(segment)
        })
    }

    fn looks_generated(segment: &str) -> bool {
        let digit = segment.bytes().any(|b| b.is_ascii_digit());
        let upper = segment.bytes().any(|b| b.is_ascii_uppercase());
        let lower = segment.bytes().any(|b| b.is_ascii_lowercase());
        digit && (upper || lower) && (segment.len() >= 16 || (upper && lower))
    }
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
            .map(|&(id, source, is_harmless)| Rule {
                id,
                regex: Regex::new(source).expect("built-in rules are valid regexes"),
                is_harmless,
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
                if (rule.is_harmless)(caps) {
                    caps[0].to_owned()
                } else {
                    Self::replace_secret(rule.id, caps)
                }
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
        if !Self::is_path(token) {
            return if Self::looks_random(token) {
                marker
            } else {
                token.to_owned()
            };
        }
        let mut start = 0;
        for segment in token.split('/') {
            let end = start + segment.len();
            if Self::looks_random(segment) {
                return format!(
                    "{}{marker}{}",
                    Self::redact_random(&token[..start]),
                    Self::redact_random(&token[end..])
                );
            }
            start = end + 1;
        }
        token.to_owned()
    }

    fn is_path(token: &str) -> bool {
        token.starts_with(['/', '~', '.'])
            || (token.contains('/')
                && token.split('/').any(|segment| {
                    segment.len() >= PATH_WORD_MIN_LEN
                        && segment.bytes().all(|b| b.is_ascii_lowercase())
                }))
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
    fn keeps_relative_paths_to_generated_directories() {
        let redactor = Redactor::with_home("/Users/ada");
        let cases = [
            (
                "cd proj/.claude/worktrees/agent-a3f9c2d17e5b4c08",
                "cd proj/.claude/worktrees/agent-a3f9c2d17e5b4c08",
            ),
            (
                "cp dist/app.js worktrees/agent-a3f9c2d17e5b4c08/dist",
                "cp dist/app.js worktrees/agent-a3f9c2d17e5b4c08/dist",
            ),
            (
                "cd /Users/ada/Library/CloudStorage/GoogleDrive-ada@example.com/shop",
                "cd ~/Library/CloudStorage/[REDACTED:email]/shop",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(redactor.redact(text), expected);
        }
    }

    #[test]
    fn detects_remaining_secrets() {
        let redactor = Redactor::with_home("/Users/ada");

        assert!(redactor.contains_secret("SECRET_TOKEN=abc123"));
        assert!(!redactor.contains_secret("cargo test -p invoicer"));
        assert!(!redactor.contains_secret("cat /Users/ada/notes.md"));
    }

    fn fake(parts: &[&str]) -> String {
        parts.concat()
    }

    fn hex() -> String {
        fake(&["4b1e9c7d2a5f8e3b", "6c0d9a2f5e8b1c4d"])
    }

    fn must_redact() -> Vec<(String, Vec<String>)> {
        let api_key = fake(&["q7Lm2Vx9", "Rt4Kp8Wz"]);
        let session = fake(&["8f2kq0z7", "xw1m"]);
        let curl_password = fake(&["Tr0ub", "4dor"]);
        let mysql_password = fake(&["S3cret", "Pw"]);
        let ssh_password = fake(&["Wint3r", "Pw"]);
        let db_password = fake(&["Kq8v", "Rz2m"]);
        let client_secret = fake(&["Zx81kQ", "p0vLm3"]);
        let app_id = fake(&["9f3c1a7e5b2d4f6a", "8c0e1b3d5f7a9c2e"]);
        let hex = hex();
        let aws_id = fake(&["AK", "IA", "Q3EGRTYN4UJ7XKZP"]);
        let aws_secret = fake(&["k3Jd9sX0pQ2mZ7vB4nR8", "tY1wE6uI5oP0aS3dF7g"]);
        let base64 = [fake(&["9j4AAQ", "SkZJRg"]), fake(&["ABAQ2w", "BDAA"])];
        let hex_path = fake(&["a1b2c3d4e5f6a7b8", "c9d0e1f2a3b4c5d6"]);
        let slashed = [
            fake(&["k3Jd9sX0pQ", "2mZ7vB4nR8"]),
            fake(&["tY1wE6uI5o", "P0aS3dF7g"]),
        ];
        let prefix = [
            fake(&["Q7vK2mXp", "9LzR4wT8"]),
            fake(&["Nb3Hy6Jc1", "Fd5Gs0Wq"]),
        ];
        let nested = fake(&["Zx81kQp0vLm3Rt4K", "p8WzB2nY6cH9sD1fG5"]);
        let pgp_body = [fake(&["lQOYBGXk", "2fIBCADq7d"]), fake(&["=pX", "3a"])];
        let pgp = [
            fake(&["-----BEGIN PGP ", "PRIVATE KEY BLOCK-----\n"]),
            pgp_body.join("\n"),
            fake(&["-----END PGP ", "PRIVATE KEY BLOCK-----"]),
        ]
        .join("\n");
        let case = |text: String, secrets: &[&str]| {
            (
                text,
                secrets.iter().map(|&secret| secret.to_owned()).collect(),
            )
        };
        vec![
            case(
                format!(r#"curl -H "X-Api-Key: {api_key}" https://api.example.com"#),
                &[&api_key],
            ),
            case(
                format!(r#"requests.get(url, headers={{"X-Api-Key": "{api_key}"}})"#),
                &[&api_key],
            ),
            case(
                format!(
                    "curl -H 'Cookie: sessionid={session}; csrftoken=Zq81' https://shop.example.com"
                ),
                &[&session, "Zq81"],
            ),
            case(
                format!("Set-Cookie: session={session}; Path=/; HttpOnly"),
                &[&session],
            ),
            case(
                r#"./scripts/seed.sh '{"password":"hunter2"}'"#.to_owned(),
                &["hunter2"],
            ),
            case(
                "{'password': 'hunter2', 'user': 'ada'}".to_owned(),
                &["hunter2"],
            ),
            case(
                "spring.datasource.password=hunter2".to_owned(),
                &["hunter2"],
            ),
            case(format!(r#"accessToken: "{api_key}.v2""#), &[&api_key]),
            case(
                format!("curl -u admin:{curl_password} https://ci.example.com/api/json"),
                &[&curl_password],
            ),
            case(
                format!("curl -sS -uadmin:{curl_password} https://ci.example.com"),
                &[&curl_password],
            ),
            case(
                format!("curl --user=admin:{curl_password} https://ci.example.com"),
                &[&curl_password],
            ),
            case(
                format!("curl -X POST --user 'admin:{curl_password}' https://ci.example.com"),
                &[&curl_password],
            ),
            case(
                format!("mysql -u root -p{mysql_password} shop"),
                &[&mysql_password],
            ),
            case(
                format!("mysqldump -h db -u root -p'{mysql_password}' shop > dump.sql"),
                &[&mysql_password],
            ),
            case(
                format!("sshpass -p {ssh_password} ssh deploy@staging"),
                &[&ssh_password],
            ),
            case(
                format!("sshpass -v -p '{ssh_password}' scp build.tar host:"),
                &[&ssh_password],
            ),
            case(
                format!("DB_PASS={db_password} ./scripts/migrate.sh"),
                &[&db_password],
            ),
            case(
                format!("PGPASS={db_password} psql -h db shop"),
                &[&db_password],
            ),
            case(format!("dbPass: {db_password}"), &[&db_password]),
            case(
                format!("./bin/oauth --client-secret {client_secret}"),
                &[&client_secret],
            ),
            case(
                format!("./bin/oauth --auth-token {client_secret} --verbose"),
                &[&client_secret],
            ),
            case(
                format!("python summarize.py --openai-api-key {client_secret}"),
                &[&client_secret],
            ),
            case("./bin/cli --password hunter2".to_owned(), &["hunter2"]),
            case("./bin/cli --token=abc123".to_owned(), &["abc123"]),
            case(
                "http --auth admin:hunter2 :8080/admin".to_owned(),
                &["admin:hunter2"],
            ),
            case(
                r#"subprocess.run(["http", "--auth=admin:hunter2", url])"#.to_owned(),
                &["hunter2"],
            ),
            case(
                "./bin/cli --verbose --token abc123 --password-stdin".to_owned(),
                &["abc123"],
            ),
            case(
                "./bin/cli --token-type bearer --token abc123".to_owned(),
                &["abc123"],
            ),
            case(
                format!(
                    r#"curl "https://api.openweathermap.org/data/2.5/weather?q=Berlin&appid={app_id}""#
                ),
                &[&app_id],
            ),
            case(
                format!(r#"curl -H "DD-API-KEY: {hex}" https://api.datadoghq.com"#),
                &[&hex],
            ),
            case(format!("export MAILGUN_KEY={hex}"), &[&hex]),
            case(format!(r#"{{"app_key": "{hex}"}}"#), &[&hex]),
            case(format!("TOKEN_TYPE={hex}"), &[&hex]),
            case(pgp, &[&pgp_body[0], &pgp_body[1]]),
            case(format!("aws s3 ls --profile é{aws_id}"), &[&aws_id]),
            case(format!("{aws_id}é"), &[&aws_id]),
            case(
                "DB_PASSWORD=123456 ./scripts/seed.sh".to_owned(),
                &["123456"],
            ),
            case("SECRET_KEY=98765".to_owned(), &["98765"]),
            case("GITHUB_TOKEN_READONLY=abc123def".to_owned(), &["abc123def"]),
            case(
                format!("AWS_SECRET_ACCESS_KEY=/{aws_secret}"),
                &[&aws_secret],
            ),
            case(
                format!("SESSION_SECRET=/{}/{}", base64[0], base64[1]),
                &[&base64[0], &base64[1]],
            ),
            case(format!("API_TOKEN=/{hex_path}"), &[&hex_path]),
            case(
                format!("sign uploads with {}/{}", slashed[0], slashed[1]),
                &[&slashed[0], &slashed[1]],
            ),
            case(
                format!("tar xf {}/{}/{nested}/src.tar", prefix[0], prefix[1]),
                &[&prefix[0], &prefix[1], &nested],
            ),
            case(
                "TOKEN_TYPE=bearer,password=hunter2".to_owned(),
                &["hunter2"],
            ),
            case("MAX_TOKENS=100,TOKEN=abc123".to_owned(), &["abc123"]),
            case("SECRET_TOKEN=abc123".to_owned(), &["abc123"]),
            case(
                "Authorization: Bearer abcdefgh123".to_owned(),
                &["abcdefgh123"],
            ),
            case(
                "psql postgres://app:hunter2@db/shop".to_owned(),
                &["hunter2"],
            ),
            case(
                "mail ada@example.com < report.txt".to_owned(),
                &["ada@example.com"],
            ),
            case(
                "ls ~/Library/CloudStorage/GoogleDrive-ada@example.com/shop".to_owned(),
                &["ada@example.com"],
            ),
        ]
    }

    fn must_keep() -> Vec<String> {
        let sha256 = hex().repeat(2);
        let sha1 = &sha256[..40];
        [
            "git clone git@github.com:org/repo.git",
            "git remote add origin git@gitlab.example.com:group/project.git",
            "git clone ssh://git@github.com/org/repo.git vendor/repo",
            "TOKENIZERS_PARALLELISM=false python scripts/embed.py",
            "python summarize.py --max-tokens=100",
            "python summarize.py --max-tokens 100",
            "MAX_TOKENS=4096 ./scripts/run.sh",
            "max_new_tokens: 512",
            "PASSWORD_FILE=/run/secrets/db ./scripts/migrate.sh",
            "./bin/cli --password-file /run/secrets/db",
            "GOOGLE_APPLICATION_CREDENTIALS=~/keys/service-account.json npm test",
            "SSL_PRIVATE_KEY=/etc/ssl/private/server.key ./bin/serve",
            "export TOKEN_PATH=./config/token.json",
            "TOKEN_TYPE=bearer",
            "TOKEN_TTL=3600 PASSWORD_MIN_LENGTH=12 ./bin/serve",
            "DEBUG_SECRETS=0 npm start",
            "USE_CREDENTIALS=true ./bin/sync",
            r#"DB_PASSWORD="" ./scripts/seed.sh"#,
            "GITHUB_TOKEN=$GITHUB_TOKEN ./scripts/release.sh",
            "npm publish --token ${NPM_TOKEN}",
            "npm config set //registry.npmjs.org/:_authToken=${NPM_TOKEN}",
            r#"curl -u "$CI_USER:$CI_PASSWORD" https://ci.example.com"#,
            r#"curl -H "Cookie: $COOKIE" https://shop.example.com"#,
            "echo $PASSWORD | docker login --username ada --password-stdin ghcr.io",
            "./bin/cli --no-password migrate",
            "assert token == expected",
            "Token::new(source).unwrap_or_default()",
            "docker run -u 1000:1000 node",
            "mysql -u root -p shop",
            "mysql -P 3306 -h db shop",
            "sshpass -f ~/.pw ssh -p 2222 deploy@staging",
            "ssh -p 2222 deploy@staging",
            "cargo test -p invoicer",
            "npm test",
            "cd .claude/worktrees/agent-a3f9c2d17e5b4c08 && cargo test",
            "diff -r worktrees/agent-a3f9c2d17e5b4c08/src src",
        ]
        .into_iter()
        .map(str::to_owned)
        .chain([
            format!("docker pull node@sha256:{sha256}"),
            format!("pip install requests --hash=sha256:{sha256}"),
            format!(r#"checksum = "{sha256}""#),
            format!(r#"{{"head_sha": "{sha1}"}}"#),
        ])
        .collect()
    }

    #[test]
    fn redacts_every_known_secret_shape() {
        let redactor = Redactor::with_home("/Users/ada");
        for (text, secrets) in must_redact() {
            let redacted = redactor.redact(&text);
            assert!(redacted.contains(MARKER_PREFIX), "{text} -> {redacted}");
            for secret in secrets {
                assert!(!redacted.contains(&secret), "{text} -> {redacted}");
            }
            assert!(redactor.contains_secret(&text), "{text}");
        }
    }

    #[test]
    fn keeps_values_that_are_not_secrets() {
        let redactor = Redactor::with_home("/Users/ada");
        for text in must_keep() {
            assert_eq!(redactor.redact(&text), text);
            assert!(!redactor.contains_secret(&text), "{text}");
        }
    }

    #[test]
    fn redacts_only_the_secret_part() {
        let redactor = Redactor::with_home("/Users/ada");
        let password = fake(&["Tr0ub", "4dor"]);
        let secret = fake(&["Zx81kQ", "p0vLm3"]);
        let cases = [
            (
                r#"{"password":"hunter2","user":"ada"}"#.to_owned(),
                r#"{"password":[REDACTED:assignment],"user":"ada"}"#,
            ),
            (
                format!("curl -u admin:{password} https://ci.example.com"),
                "curl -u admin:[REDACTED:cli-password] https://ci.example.com",
            ),
            (
                format!("./bin/oauth --client-secret {secret} --verbose"),
                "./bin/oauth --client-secret [REDACTED:flag] --verbose",
            ),
            (
                "git clone git@github.com:org/repo.git && mail ada@example.com".to_owned(),
                "git clone git@github.com:org/repo.git && mail [REDACTED:email]",
            ),
        ];
        for (text, expected) in cases {
            assert_eq!(redactor.redact(&text), expected);
        }
    }

    #[test]
    fn redacting_twice_changes_nothing() {
        let redactor = Redactor::with_home("/Users/ada");
        let texts = must_redact()
            .into_iter()
            .map(|(text, _)| text)
            .chain(must_keep());
        for text in texts {
            let once = redactor.redact(&text).into_owned();
            assert_eq!(redactor.redact(&once), once);
        }
    }

    quickcheck! {
        fn redaction_is_idempotent(text: String) -> bool {
            let redactor = Redactor::with_home("/Users/ada");
            let once = redactor.redact(&text).into_owned();
            redactor.redact(&once) == once
        }
    }
}
