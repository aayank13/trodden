use unicode_normalization::{UnicodeNormalization, char::is_combining_mark};

const MAX_WORD_CHARS: usize = 100;

const CONTINUATION: &str = "##";

pub(crate) trait Vocabulary {
    fn id(&self, token: &str) -> Option<u32>;
}

#[derive(Debug)]
pub(crate) struct Tokenizer;

impl Tokenizer {
    pub(crate) fn encode(text: &str, vocab: &impl Vocabulary) -> Vec<u32> {
        let normalized = Self::normalize(text);
        let mut ids = Vec::new();
        for word in Self::pre_tokenize(&normalized) {
            Self::word_piece(word, vocab, &mut ids);
        }
        ids
    }

    fn normalize(text: &str) -> String {
        let mut cleaned = String::with_capacity(text.len());
        for c in text.chars() {
            if c == '\0' || c == '\u{fffd}' || (c.is_control() && !c.is_whitespace()) {
                continue;
            }
            if c.is_whitespace() {
                cleaned.push(' ');
            } else if Self::is_cjk(c) {
                cleaned.push(' ');
                cleaned.push(c);
                cleaned.push(' ');
            } else {
                cleaned.push(c);
            }
        }
        cleaned
            .nfd()
            .filter(|c| !is_combining_mark(*c))
            .flat_map(char::to_lowercase)
            .collect()
    }

    fn pre_tokenize(text: &str) -> Vec<&str> {
        let mut words = Vec::new();
        for chunk in text.split_whitespace() {
            let mut start = 0;
            for (index, c) in chunk.char_indices() {
                if Self::is_punctuation(c) {
                    if start < index {
                        words.push(&chunk[start..index]);
                    }
                    words.push(&chunk[index..index + c.len_utf8()]);
                    start = index + c.len_utf8();
                }
            }
            if start < chunk.len() {
                words.push(&chunk[start..]);
            }
        }
        words
    }

    fn word_piece(word: &str, vocab: &impl Vocabulary, ids: &mut Vec<u32>) {
        if word.chars().count() > MAX_WORD_CHARS {
            return;
        }
        let mut pieces = Vec::new();
        let mut piece = String::new();
        let mut start = 0;
        while start < word.len() {
            let mut end = word.len();
            let found = loop {
                piece.clear();
                if start > 0 {
                    piece.push_str(CONTINUATION);
                }
                piece.push_str(&word[start..end]);
                if let Some(id) = vocab.id(&piece) {
                    break Some(id);
                }
                match word[start..end].char_indices().last() {
                    Some((offset, _)) if offset > 0 => end = start + offset,
                    _ => break None,
                }
            };
            let Some(id) = found else { return };
            pieces.push(id);
            start = end;
        }
        ids.extend(pieces);
    }

    fn is_punctuation(c: char) -> bool {
        c.is_ascii_punctuation()
            || matches!(c, '\u{2010}'..='\u{2027}' | '\u{3000}'..='\u{303f}' | '¡' | '¿' | '«' | '»')
    }

    fn is_cjk(c: char) -> bool {
        matches!(c,
            '\u{4e00}'..='\u{9fff}'
            | '\u{3400}'..='\u{4dbf}'
            | '\u{20000}'..='\u{2a6df}'
            | '\u{2a700}'..='\u{2b73f}'
            | '\u{2b740}'..='\u{2b81f}'
            | '\u{2b820}'..='\u{2ceaf}'
            | '\u{f900}'..='\u{faff}'
            | '\u{2f800}'..='\u{2fa1f}')
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct Tokens(HashMap<&'static str, u32>);

    impl Vocabulary for Tokens {
        fn id(&self, token: &str) -> Option<u32> {
            self.0.get(token).copied()
        }
    }

    #[test]
    fn splits_words_into_known_pieces() {
        let vocab = Tokens(HashMap::from([
            ("pag", 1),
            ("##inate", 2),
            ("(", 3),
            (")", 4),
            ("items", 5),
            ("cafe", 6),
        ]));

        assert_eq!(
            Tokenizer::encode("Paginate(items) Café", &vocab),
            [1, 2, 3, 5, 4, 6]
        );
    }

    #[test]
    fn drops_words_with_unknown_pieces() {
        let vocab = Tokens(HashMap::from([("fix", 1), ("it", 2)]));

        assert_eq!(Tokenizer::encode("fix zzz it", &vocab), [1, 2]);
    }
}
