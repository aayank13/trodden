use std::{iter, ops::RangeInclusive};

use unicode_normalization::UnicodeNormalization;

const TERM_CHARS: [char; 4] = ['_', '-', '.', '/'];

const DIACRITICS: RangeInclusive<char> = '\u{300}'..='\u{36F}';

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
        Self::unaccented(text)
            .to_lowercase()
            .split(|c: char| !(c.is_alphanumeric() || TERM_CHARS.contains(&c)))
            .map(|term| term.trim_matches(|c: char| TERM_CHARS.contains(&c)))
            .filter(|term| !term.is_empty() && !STOP_WORDS.contains(term))
            .map(str::to_owned)
            .collect()
    }

    pub fn normalize(text: &str) -> String {
        Self::of(text).join(" ")
    }

    fn unaccented(text: &str) -> String {
        text.chars()
            .filter(|c| !DIACRITICS.contains(c))
            .map(Self::base_letter)
            .collect()
    }

    fn base_letter(letter: char) -> char {
        if letter.is_ascii() {
            return letter;
        }
        let mut parts = iter::once(letter).nfd();
        match (parts.next(), parts.next(), parts.next()) {
            (Some(base), Some(mark), None)
                if base.is_ascii_alphabetic() && DIACRITICS.contains(&mark) =>
            {
                base
            }
            _ => letter,
        }
    }
}

#[cfg(test)]
mod tests {
    use trodden_core::Procedure;

    use super::*;
    use crate::Store;

    const REPO: &str = "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c";

    const ACCENTED: [(&str, &str); 7] = [
        ("café", "cafe"),
        ("naïve", "naive"),
        ("résumé", "resume"),
        ("Zürich", "zurich"),
        ("São", "sao"),
        ("Ångström", "angstrom"),
        ("crème brûlée", "creme brulee"),
    ];

    #[derive(Debug)]
    struct Indexed(Store);

    impl Indexed {
        fn titled(title: &str) -> Self {
            let mut store = Store::open_in_memory().expect("store opens");
            let mut procedure = Procedure::example();
            procedure.title = title.to_owned();
            store.upsert(&procedure).expect("stored");
            Self(store)
        }

        fn raw(text: &str) -> Self {
            let store = Store::open_in_memory().expect("store opens");
            store
                .conn
                .execute(
                    "INSERT INTO procedures_fts (rowid, title, body) VALUES (1, ?1, '')",
                    [text],
                )
                .expect("text indexed");
            Self(store)
        }

        fn vocabulary(&self) -> Vec<String> {
            self.0
                .conn
                .prepare("SELECT term FROM procedures_vocab ORDER BY term")
                .expect("vocabulary query prepares")
                .query_map([], |row| row.get(0))
                .expect("vocabulary reads")
                .collect::<Result<_, _>>()
                .expect("vocabulary rows read")
        }

        fn finds(&self, prompt: &str) -> bool {
            !self
                .0
                .lexical_hits(&Terms::of(prompt), REPO, 5)
                .expect("text searched")
                .is_empty()
        }
    }

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

    #[test]
    fn strips_accents_from_composed_and_decomposed_text() {
        for (accented, plain) in ACCENTED {
            let decomposed: String = accented.nfd().collect();
            assert_eq!(Terms::normalize(accented), plain, "{accented}");
            assert_eq!(Terms::normalize(&decomposed), plain, "{accented}");
        }
    }

    #[test]
    fn keeps_letters_the_index_keeps() {
        assert_eq!(
            Terms::of("Straße smørrebrød Ελληνικά Йошкар-Ола ベージ"),
            ["straße", "smørrebrød", "ελληνικά", "йошкар-ола", "ベージ"],
        );
    }

    #[test]
    fn folds_text_the_way_the_index_does() {
        let words = ACCENTED.iter().map(|(accented, _)| *accented).chain([
            "İstanbul",
            "Ελληνικά",
            "Tiếng Việt",
            "Йошкар-Ола",
            "ǖ ǣ ǿ",
        ]);
        for word in words {
            let mut terms = Terms::of(word);
            terms.sort();
            assert_eq!(Indexed::raw(word).vocabulary(), terms, "{word}");
        }
    }

    #[test]
    fn accented_and_plain_words_find_each_other() {
        for (accented, plain) in ACCENTED {
            for (title, prompt) in [
                (accented, accented),
                (accented, plain),
                (plain, accented),
                (plain, plain),
            ] {
                assert!(
                    Indexed::titled(title).finds(prompt),
                    "{prompt} finds {title}"
                );
            }
        }
    }
}
