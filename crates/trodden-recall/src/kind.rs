use crate::Skeleton;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskKind {
    Fix,
    Add,
    Meta,
}

impl TaskKind {
    const FIX: &[&str] = &[
        "fix",
        "bug",
        "broken",
        "breaks",
        "crash",
        "crashes",
        "fails",
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

    const META: &[&str] = &[
        "document",
        "describe",
        "explain",
        "docstring",
        "docstrings",
        "readme",
        "test",
        "tests",
        "comment",
        "comments",
    ];

    pub fn of(prompt: &str) -> Option<Self> {
        let text = Skeleton::of(prompt).as_str().to_lowercase();
        let words: Vec<&str> = text
            .split(|c: char| !(c.is_alphanumeric() || matches!(c, '\'' | '-' | '_')))
            .filter(|word| !word.is_empty())
            .collect();
        let has = |list: &[&str]| words.iter().any(|word| list.contains(word));
        if has(Self::META) {
            Some(Self::Meta)
        } else if has(Self::FIX) || Self::FIX_PHRASES.iter().any(|phrase| text.contains(phrase)) {
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
}
