use std::sync::LazyLock;

use regex::Regex;

const MAX_SYMBOLS: usize = 5;

const MAX_LOOKBACK_LINES: usize = 200;

static DEFINITION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (?:export\s+(?:default\s+)?)?
        (?:pub(?:\([^)]*\))?\s+)?
        (?:async\s+|unsafe\s+|static\s+|const\s+|extern\s+(?:\x22[^\x22]*\x22\s+)?)*
        (?:fn|def|class|struct|enum|trait|union|mod|interface|func|function|impl(?:<[^>]*>)?|type)
        \s+
        (?:\([^)]*\)\s*)?
        (?:[A-Za-z_][A-Za-z0-9_:<>,\ ]*\s+for\s+)?
        (?P<name>[A-Za-z_$][A-Za-z0-9_$]*)
        ",
    )
    .expect("definition pattern is valid")
});

static CONSTANT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:const|static)\s+(?:mut\s+)?(?P<name>[A-Z][A-Z0-9_]*)\s*:",
    )
    .expect("constant pattern is valid")
});

static ARROW_DEFINITION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s+)?(?:\([^)]*\)\s*=>|[A-Za-z_$][A-Za-z0-9_$]*\s*=>|function\b)",
    )
    .expect("arrow definition pattern is valid")
});

#[derive(Debug, Clone, Copy)]
pub(crate) struct Hunk {
    pub(crate) old_start: usize,
}

#[derive(Debug)]
pub(crate) struct SymbolFinder;

impl SymbolFinder {
    pub(crate) fn find<'a>(
        changed_lines: impl IntoIterator<Item = &'a str>,
        original: Option<&str>,
        hunks: &[Hunk],
    ) -> Vec<String> {
        let mut symbols = Vec::new();
        let mut push = |name: &str| {
            if symbols.len() < MAX_SYMBOLS && !symbols.iter().any(|s: &String| s == name) {
                symbols.push(name.to_owned());
            }
        };

        for line in changed_lines {
            if let Some(name) = Self::definition(line) {
                push(name);
            }
        }

        if let Some(original) = original {
            let lines: Vec<&str> = original.lines().collect();
            for hunk in hunks.iter().filter(|_| !lines.is_empty()) {
                let start = hunk.old_start.clamp(1, lines.len()) - 1;
                let enclosing = lines[..=start]
                    .iter()
                    .rev()
                    .take(MAX_LOOKBACK_LINES)
                    .find_map(|line| Self::definition(line));
                if let Some(name) = enclosing {
                    push(name);
                }
            }
        }
        symbols
    }

    fn definition(line: &str) -> Option<&str> {
        DEFINITION
            .captures(line)
            .or_else(|| CONSTANT.captures(line))
            .or_else(|| ARROW_DEFINITION.captures(line))
            .and_then(|caps| caps.name("name"))
            .map(|name| name.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_definitions_in_many_languages() {
        let lines = [
            "pub(crate) async fn load_all(path: &Path) -> Result<Vec<Invoice>> {",
            "def order_total_cents(order, product):",
            "export function paginate(items, page, perPage) {",
            "impl fmt::Display for Invoice {",
            "class Customer:",
            "export const listProducts = ({ page = 1 } = {}) => {",
            "func (s *Server) Handle(w http.ResponseWriter) {",
            "pub const NAMES: &[&str] = &[\"list\", \"show\", \"help\"];",
        ];
        let names: Vec<_> = lines
            .iter()
            .filter_map(|line| SymbolFinder::definition(line))
            .collect();

        assert_eq!(
            names,
            [
                "load_all",
                "order_total_cents",
                "paginate",
                "Invoice",
                "Customer",
                "listProducts",
                "Handle",
                "NAMES"
            ]
        );
    }

    #[test]
    fn finds_the_function_enclosing_a_hunk() {
        let original = "import x\n\nexport function paginate(items, page, perPage) {\n  const start = (page - 1) * perPage;\n  return items.slice(start, start + perPage + 1);\n}\n";

        let symbols = SymbolFinder::find(
            ["  return items.slice(start, start + perPage);"],
            Some(original),
            &[Hunk { old_start: 5 }],
        );

        assert_eq!(symbols, ["paginate"]);
    }
}
