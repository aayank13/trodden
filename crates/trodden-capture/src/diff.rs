#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Diff {
    pub(crate) starts: Vec<usize>,
    pub(crate) changed: Vec<String>,
    pub(crate) added: u32,
    pub(crate) removed: u32,
}

impl Diff {
    pub(crate) fn line(&mut self, line: &str) {
        if let Some(text) = line.strip_prefix('+') {
            self.added = self.added.saturating_add(1);
            self.changed.push(text.to_owned());
        } else if let Some(text) = line.strip_prefix('-') {
            self.removed = self.removed.saturating_add(1);
            self.changed.push(text.to_owned());
        }
    }

    pub(crate) fn unified(text: &str) -> Self {
        let mut diff = Self::default();
        let mut in_hunk = !text.lines().any(|line| line.starts_with("@@"));
        for line in text.lines() {
            if let Some(header) = line.strip_prefix("@@") {
                in_hunk = true;
                if let Some(start) = Self::old_start(header) {
                    diff.starts.push(start);
                }
            } else if in_hunk {
                diff.line(line);
            }
        }
        diff
    }

    fn old_start(header: &str) -> Option<usize> {
        let old = header.trim_start().strip_prefix('-')?;
        let digits: String = old.chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    }

    pub(crate) fn created(content: &str) -> Self {
        let changed: Vec<String> = content.lines().map(str::to_owned).collect();
        Self {
            starts: Vec::new(),
            added: u32::try_from(changed.len()).unwrap_or(u32::MAX),
            changed,
            removed: 0,
        }
    }

    pub(crate) fn replaced(old: &str, new: &str) -> Self {
        let old: Vec<&str> = old.lines().collect();
        let new: Vec<&str> = new.lines().collect();
        let prefix = old
            .iter()
            .zip(&new)
            .take_while(|(old, new)| old == new)
            .count();
        let suffix = old[prefix..]
            .iter()
            .rev()
            .zip(new[prefix..].iter().rev())
            .take_while(|(old, new)| old == new)
            .count();
        let removed = &old[prefix..old.len() - suffix];
        let added = &new[prefix..new.len() - suffix];
        Self {
            starts: Vec::new(),
            changed: removed
                .iter()
                .chain(added)
                .map(|line| (*line).to_owned())
                .collect(),
            added: u32::try_from(added.len()).unwrap_or(u32::MAX),
            removed: u32::try_from(removed.len()).unwrap_or(u32::MAX),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_diffs_skip_file_headers_and_keep_hunk_starts() {
        let diff = Diff::unified(
            "--- a/src/paginate.js\n+++ b/src/paginate.js\n@@ -3,4 +3,4 @@ export function paginate\n   const start = (page - 1) * perPage;\n-  const end = start + perPage + 1;\n+  const end = start + perPage;\n   return items.slice(start, end);\n@@ -20 +20,2 @@\n+// paging\n",
        );

        assert_eq!(diff.starts, [3, 20]);
        assert_eq!(diff.added, 2);
        assert_eq!(diff.removed, 1);
        assert_eq!(
            diff.changed,
            [
                "  const end = start + perPage + 1;",
                "  const end = start + perPage;",
                "// paging"
            ]
        );
    }

    #[test]
    fn patch_bodies_without_line_numbers_still_count_lines() {
        let diff = Diff::unified("@@ fn total\n-    sum\n+    sum + tax\n");

        assert!(diff.starts.is_empty());
        assert_eq!((diff.added, diff.removed), (1, 1));
    }

    #[test]
    fn bodies_without_hunk_headers_are_all_hunk_lines() {
        let diff = Diff::unified("+fn main() {}\n+\n");

        assert_eq!((diff.added, diff.removed), (2, 0));
    }

    #[test]
    fn replacements_drop_the_context_both_sides_share() {
        let diff = Diff::replaced(
            "fn total() {\n    sum\n}\n",
            "fn total() {\n    sum + tax\n    // rounded\n}\n",
        );

        assert_eq!((diff.added, diff.removed), (2, 1));
        assert_eq!(diff.changed, ["    sum", "    sum + tax", "    // rounded"]);
    }

    #[test]
    fn created_files_add_every_line() {
        let diff = Diff::created("a\nb\nc");

        assert_eq!((diff.added, diff.removed), (3, 0));
        assert_eq!(diff.changed, ["a", "b", "c"]);
    }
}
