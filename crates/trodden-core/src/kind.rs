use crate::Skeleton;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    Fix,
    Add,
    Meta,
}

impl TaskKind {
    pub const ALL: [Self; 3] = [Self::Fix, Self::Add, Self::Meta];

    const FIX: &[&str] = &[
        "fix",
        "bug",
        "broken",
        "breaks",
        "crash",
        "crashes",
        "fail",
        "fails",
        "failed",
        "failing",
        "error",
        "errors",
        "wrong",
        "incorrect",
        "instead",
        "repeats",
        "duplicate",
        "duplicates",
        "overflow",
        "overflows",
        "overwrites",
        "skip",
        "skips",
        "forever",
        "behind",
        "regression",
    ];

    const FIX_PHRASES: &[&str] = &[
        "off by one",
        "off-by-one",
        "doesn't work",
        "does not work",
        "no longer",
        "too few",
        "too many",
        "also returns",
        "returns nothing",
        "is empty when",
    ];

    const ADD: &[&str] = &[
        "add",
        "adds",
        "adding",
        "create",
        "introduce",
        "implement",
        "support",
        "new",
        "need",
        "needs",
        "give",
        "expose",
        "want",
    ];

    const DOCS: &[&str] = &[
        "document",
        "describe",
        "explain",
        "docstring",
        "docstrings",
        "readme",
        "comment",
        "comments",
    ];

    const TESTS: &[&str] = &["test", "tests"];

    const WRITE: &[&str] = &[
        "add", "adds", "adding", "write", "writes", "writing", "create", "cover", "covers",
        "covering", "extend",
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fix => "fix",
            Self::Add => "add",
            Self::Meta => "meta",
        }
    }

    pub fn of(prompt: &str) -> Option<Self> {
        let text = Skeleton::of(prompt)
            .as_str()
            .replace(['\u{2018}', '\u{2019}'], "'")
            .to_lowercase();
        let words: Vec<&str> = text
            .split(|c: char| !(c.is_alphanumeric() || matches!(c, '\'' | '-' | '_')))
            .filter(|word| !word.is_empty())
            .collect();
        let has = |list: &[&str]| words.iter().any(|word| list.contains(word));
        let symptom =
            has(Self::FIX) || Self::FIX_PHRASES.iter().any(|phrase| text.contains(phrase));
        if has(Self::DOCS) {
            Some(Self::Meta)
        } else if words.contains(&"fix") {
            Some(Self::Fix)
        } else if has(Self::TESTS) && (has(Self::WRITE) || !symptom) {
            Some(Self::Meta)
        } else if symptom {
            Some(Self::Fix)
        } else if has(Self::ADD) {
            Some(Self::Add)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tells_symptoms_from_requests() {
        let cases = [
            (
                "Page 2 of the product listing repeats the last product from page 1. Fix it.",
                Some(TaskKind::Fix),
            ),
            (
                "Listing a project's open tasks also returns tasks that were already completed.",
                Some(TaskKind::Fix),
            ),
            (
                "Tasks need a priority. Add an integer column to tasks.",
                Some(TaskKind::Add),
            ),
            (
                "Could you give invoicer a subcommand listing invoices due this month?",
                Some(TaskKind::Add),
            ),
            (
                "Add a unit test for formatPrice with the fr locale.",
                Some(TaskKind::Meta),
            ),
            (
                "Document the listing functions in the README.",
                Some(TaskKind::Meta),
            ),
            ("perPage values above 100 should be capped at 100.", None),
            ("create_task accepts an empty title.", None),
            ("Rename `add_item` to `append`.", None),
        ];
        for (prompt, kind) in cases {
            assert_eq!(TaskKind::of(prompt), kind, "{prompt}");
        }
    }

    #[test]
    fn failing_tests_are_fixes() {
        let cases = [
            "The tests fail because page 2 repeats the last item",
            "Fix the failing test in paginate",
            "Page 2 repeats the last product from page 1, fix it and add a test",
            "The test for parse_date has been failing since the timezone change",
            "The checkout test fails on CI with a timeout",
            "npm test reports 3 failed tests in cart.spec.js",
            "Our integration tests crash when the database is empty",
            "Running the tests gives a duplicate key error",
        ];
        for prompt in cases {
            assert_eq!(TaskKind::of(prompt), Some(TaskKind::Fix), "{prompt}");
        }
    }

    #[test]
    fn reads_curly_apostrophes() {
        assert_eq!(
            TaskKind::of("Paging doesn\u{2019}t work after the refactor"),
            Some(TaskKind::Fix)
        );
        assert_eq!(
            TaskKind::of("Paging doesn't work after the refactor"),
            Some(TaskKind::Fix)
        );
    }

    #[test]
    fn keeps_other_prompts_in_their_kind() {
        let cases = [
            (
                "The export crashes when a row has no date",
                Some(TaskKind::Fix),
            ),
            (
                "Totals are off by one when the cart is empty",
                Some(TaskKind::Fix),
            ),
            (
                "Search returns nothing for queries with accents",
                Some(TaskKind::Fix),
            ),
            ("Fix the typo in the error message", Some(TaskKind::Fix)),
            (
                "Add a dark mode toggle to the settings page",
                Some(TaskKind::Add),
            ),
            ("Implement CSV export for invoices", Some(TaskKind::Add)),
            (
                "We need a retry option on the upload command",
                Some(TaskKind::Add),
            ),
            ("Support filtering tasks by tag", Some(TaskKind::Add)),
            (
                "Write tests for the error handling in parse_config",
                Some(TaskKind::Meta),
            ),
            (
                "Add a regression test for the off-by-one in paginate",
                Some(TaskKind::Meta),
            ),
            (
                "Add a test that fails when page 2 repeats the last item",
                Some(TaskKind::Meta),
            ),
            ("Cover the empty cart case with tests", Some(TaskKind::Meta)),
            (
                "Increase test coverage for the billing module",
                Some(TaskKind::Meta),
            ),
            (
                "Add tests and a new endpoint for refunds",
                Some(TaskKind::Meta),
            ),
            (
                "Document the errors returned by parse",
                Some(TaskKind::Meta),
            ),
            (
                "Explain why the build fails on Windows",
                Some(TaskKind::Meta),
            ),
            ("Fix the docstring of paginate", Some(TaskKind::Meta)),
            (
                "Update the README with the new install steps",
                Some(TaskKind::Meta),
            ),
            (
                "Add comments explaining the overflow check",
                Some(TaskKind::Meta),
            ),
            ("Bump the tokio version", None),
        ];
        for (prompt, kind) in cases {
            assert_eq!(TaskKind::of(prompt), kind, "{prompt}");
        }
    }
}
