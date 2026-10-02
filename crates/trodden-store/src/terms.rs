const TERM_CHARS: [char; 4] = ['_', '-', '.', '/'];

const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "do", "for", "from", "has",
    "have", "i", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or", "our",
    "please", "so", "that", "the", "their", "them", "then", "there", "this", "to", "was", "we",
    "when", "which", "with", "you", "your",
];

#[derive(Debug)]
pub struct Terms;

impl Terms {
    pub fn of(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !(c.is_alphanumeric() || TERM_CHARS.contains(&c)))
            .map(|term| term.trim_matches(|c: char| TERM_CHARS.contains(&c)))
            .filter(|term| !term.is_empty() && !STOP_WORDS.contains(term))
            .map(str::to_owned)
            .collect()
    }

    pub fn normalize(text: &str) -> String {
        Self::of(text).join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_paths_whole_and_drops_stop_words() {
        assert_eq!(
            Terms::of("Fix `src/paginate.js`: page 2 repeats the last item."),
            [
                "fix",
                "src/paginate.js",
                "page",
                "2",
                "repeats",
                "last",
                "item"
            ],
        );
    }
}
