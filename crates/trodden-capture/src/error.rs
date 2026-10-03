use std::sync::LazyLock;

use regex::Regex;
use trodden_redact::Redactor;

const MAX_CHARS: usize = 120;

static PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"^(?:Exception in thread "[^"]*"|Uncaught|(?P<path>(?:[^\s:()'"`]*/)?[^\s:()'"`/]+\.[A-Za-z0-9]+)(?:(?P<column>:\d+:\d+(?::| -)|\(\d+,\d+\):)|:\d+:))\s+"#,
    )
    .expect("location prefix pattern is valid")
});

static KINDED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?:error(?:\[E\d+\]| TS\d+)?|[A-Za-z_.]*(?:Error|Exception)(?: \[[A-Z_]+\])?|panic|fatal(?: error)?):\s*\S|npm (?:ERR!|error)\s+\S)")
        .expect("error kind pattern is valid")
});

static BARE_KIND: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z_][\w.]*(?:Error|Exception)$").expect("bare kind pattern is valid")
});

static NOT_AN_ERROR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^(?:warning|note|info|hint|help|remark)\b|^npm (?:ERR!|error) (?:code|errno|syscall|path|dest)\b")
        .expect("non-error pattern is valid")
});

static PLAIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)command not found|no such file or directory|cannot find module|not recognized as|permission denied|module not found|unresolved import|undefined reference")
        .expect("plain error pattern is valid")
});

static ASSERTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\S*assert|\bassert(?:ion)?\b|\bexpect\(|\bto (?:strictly )?(?:deep-?)?equal\b|\btests? failed\b|\bfailures?=")
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
        let kinded = if traceback {
            lines.iter().rev().find_map(|line| Self::kinded(line))
        } else {
            lines.iter().find_map(|line| Self::kinded(line))
        };
        let line = kinded.or_else(|| {
            lines
                .iter()
                .copied()
                .find(|line| PLAIN.is_match(line) && !ASSERTION.is_match(line))
        })?;
        Some(Self::normalize(line))
    }

    fn kinded(line: &str) -> Option<&str> {
        let (message, located, column) = match PREFIX.captures(line) {
            Some(found) => (
                &line[found.get(0).expect("a match has a whole span").end()..],
                found.name("path").is_some(),
                found.name("column").is_some(),
            ),
            None => (line, false, false),
        };
        let error = KINDED.is_match(message) || (!located && BARE_KIND.is_match(message)) || column;
        (error && !NOT_AN_ERROR.is_match(message) && !ASSERTION.is_match(message))
            .then_some(message)
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
    fn recognizes_compiler_and_runtime_errors_whatever_their_wording() {
        let cases = [
            (
                "error[E0308]: mismatched types\n --> src/main.rs:2:22\n  |\n2 |     let total: u32 = \"3\";\n  |                ---   ^^^ expected `u32`, found `&str`",
                "error[e0308]: mismatched types",
            ),
            (
                "   Compiling shop v0.1.0 (/home/dev/shop)\nwarning: unused variable: `x`\nerror[E0599]: no method named `totl` found for struct `Cart` in the current scope\nerror: could not compile `shop` (bin \"shop\") due to 1 previous error",
                "error[e0599]: no method named `totl` found for struct `cart` in the current scope",
            ),
            (
                "error: expected one of `,`, `:`, or `}`, found `{`",
                "error: expected one of `,`, `:`, or `}`, found `{`",
            ),
            (
                "error[E0277]: the trait bound `Total: Serialize` is not satisfied",
                "error[e0277]: the trait bound `total: serialize` is not satisfied",
            ),
            (
                "SyntaxError: Unexpected token '}'",
                "syntaxerror: unexpected token '}'",
            ),
            (
                "Traceback (most recent call last):\n  File \"/home/dev/app/io.py\", line 9, in <module>\n    open(None)\nTypeError: expected str, bytes or os.PathLike object, not NoneType",
                "typeerror: expected str, bytes or os.pathlike object, not nonetype",
            ),
            (
                "  File \"/home/dev/app/cli.py\", line 3\n    def main(\n            ^\nSyntaxError: '(' was never closed",
                "syntaxerror: '(' was never closed",
            ),
            (
                "Traceback (most recent call last):\n  File \"/home/dev/app/cli.py\", line 3, in <module>\n    raise ConfigMissing\nshop.errors.ConfigMissingError",
                "shop.errors.configmissingerror",
            ),
            (
                "# example.com/shop\n./main.go:5:2: undefined: totalPrice\n./main.go:9:7: \"fmt\" imported and not used",
                "undefined: totalprice",
            ),
            (
                "internal/cart/cart.go:41:12: cannot use price (variable of type float64) as int value in return statement",
                "cannot use price (variable of type float64) as int value in return statement",
            ),
            (
                "panic: runtime error: index out of range [5] with length 3\n\ngoroutine 1 [running]:\nmain.main()\n\t/home/dev/shop/main.go:8 +0x1d",
                "panic: runtime error: index out of range [N] with length N",
            ),
            (
                "src/cart.ts(3,5): error TS2322: Type 'string' is not assignable to type 'number'.",
                "error ts2322: type 'string' is not assignable to type 'number'.",
            ),
            (
                "src/cart.ts:3:5 - error TS2322: Type 'string' is not assignable to type 'number'.",
                "error ts2322: type 'string' is not assignable to type 'number'.",
            ),
            (
                "error TS5058: The specified path does not exist: 'tsconfig.app.json'.",
                "error ts5058: the specified path does not exist: 'tsconfig.app.json'.",
            ),
            (
                "/home/dev/shop/index.js:3\nconst id = order.id;\n                 ^\n\nTypeError: Cannot read properties of undefined (reading 'id')\n    at Object.<anonymous> (/home/dev/shop/index.js:3:18)",
                "typeerror: cannot read properties of undefined (reading 'id')",
            ),
            (
                "node:internal/fs/utils:347\n    throw err;\nTypeError [ERR_INVALID_ARG_TYPE]: The \"path\" argument must be of type string. Received undefined",
                "typeerror [err_invalid_arg_type]: the \"path\" argument must be of type string. received undefined",
            ),
            (
                "Uncaught ReferenceError: process is not defined",
                "referenceerror: process is not defined",
            ),
            (
                "npm ERR! Missing script: \"build\"\nnpm ERR!\nnpm ERR! To see a list of scripts, run:\nnpm ERR!   npm run",
                "npm err! missing script: \"build\"",
            ),
            (
                "npm ERR! code E404\nnpm ERR! 404 Not Found - GET https://registry.npmjs.org/left-padd - Not found",
                "npm err! N not found - get https:<path> - not found",
            ),
            (
                "npm error code ERESOLVE\nnpm error ERESOLVE unable to resolve dependency tree",
                "npm error eresolve unable to resolve dependency tree",
            ),
            (
                "Exception in thread \"main\" java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"name\" is null\n\tat Shop.main(Shop.java:5)",
                "java.lang.nullpointerexception: cannot invoke \"string.length()\" because \"name\" is null",
            ),
            (
                "Exception in thread \"main\" java.lang.IllegalStateException\n\tat Shop.main(Shop.java:5)",
                "java.lang.illegalstateexception",
            ),
            (
                "Shop.java:5: error: cannot find symbol\n    total = prise * 2;\n            ^\n  symbol:   variable prise",
                "error: cannot find symbol",
            ),
            (
                "cart.c: In function 'main':\ncart.c:3:9: warning: unused variable 'n' [-Wunused-variable]\ncart.c:5:5: error: use of undeclared identifier 'totl'",
                "error: use of undeclared identifier 'totl'",
            ),
            (
                "cart.c:1:10: fatal error: 'shop.h' file not found\n#include \"shop.h\"\n         ^~~~~~~~",
                "fatal error: 'shop.h' file not found",
            ),
            (
                "/usr/bin/ld: /tmp/ccq1.o: in function `main':\ncart.c:(.text+0x9): undefined reference to `total'\ncollect2: error: ld returned 1 exit status",
                "cart.c:(.text+<hex>): undefined reference to `total'",
            ),
            (
                "shop/cart.py:12: error: Incompatible return value type (got \"str\", expected \"int\")  [return-value]\nFound 1 error in 1 file (checked 3 source files)",
                "error: incompatible return value type (got \"str\", expected \"int\") [return-value]",
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
    fn signatures_stay_the_same_wherever_the_error_is() {
        let at = |path: &str, line: u32| {
            ErrorSignature::of(
                &format!("{path}:{line}:5: error: use of undeclared identifier 'totl'"),
                &redactor(),
            )
        };
        assert_eq!(at("cart.c", 3), at("/home/ada/shop/src/cart.c", 41));
        assert_eq!(
            ErrorSignature::of(
                "src/cart.ts(3,5): error TS2322: Type 'string' is not assignable to type 'number'.",
                &redactor()
            ),
            ErrorSignature::of(
                "/home/ada/shop/src/lib/total.ts(88,13): error TS2322: Type 'string' is not assignable to type 'number'.",
                &redactor()
            ),
        );
    }

    #[test]
    fn ignores_test_assertions_and_unrecognized_output() {
        for output in [
            "AssertionError [ERR_ASSERTION]: Expected values to be strictly deep-equal:",
            "FAILED (failures=1)",
            "Exit code 1\n3 passing, 1 failing",
            "Traceback (most recent call last):\n  File \"/home/dev/shop/test_cart.py\", line 8, in test_total\n    self.assertEqual(total(), 4)\nAssertionError: 3 != 4",
            "    def test_total():\n>       assert total([1, 2]) == 4\nE       assert 3 == 4\nE        +  where 3 = total([1, 2])\n\ntest_cart.py:5: AssertionError\n=========================== short test summary info ============================\nFAILED test_cart.py::test_total - assert 3 == 4",
            "    def test_parse():\n>       int(\"x\")\nE       ValueError: invalid literal for int() with base 10: 'x'\n\ntest_cart.py:4: ValueError",
            "  ● total › adds the discount\n\n    expect(received).toBe(expected) // Object.is equality\n\n    Expected: 4\n    Received: 3\n\n      at Object.<anonymous> (src/cart.test.js:5:17)\n\nTests:       1 failed, 1 total",
            "Error: expect(received).toEqual(expected) // deep equality",
            "AssertionError: expected 3 to deeply equal 4",
            "JestAssertionError: expect(received).toBe(expected)",
            "running 1 test\ntest tests::adds ... FAILED\n\nfailures:\n\n---- tests::adds stdout ----\nthread 'tests::adds' panicked at src/lib.rs:10:5:\nassertion `left == right` failed\n  left: 3\n right: 4\n\ntest result: FAILED. 0 passed; 1 failed\n\nerror: test failed, to rerun pass `--lib`",
            "--- FAIL: TestTotal (0.00s)\n    cart_test.go:8: expected 4, got 3\nFAIL\nFAIL\texample.com/shop\t0.003s",
            "src/cart.rs:3:5: warning: unused variable: `x`",
            "npm ERR! code ELIFECYCLE\nnpm ERR! errno 1",
            "src/lib.rs:10:5",
            "Error",
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
