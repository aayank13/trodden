#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skeleton(String);

impl Skeleton {
    pub fn of(text: &str) -> Self {
        let mut skeleton = String::with_capacity(text.len());
        let mut rest = text;
        while let Some((start, open)) = rest
            .char_indices()
            .find(|(_, c)| matches!(c, '`' | '"' | '\u{201c}'))
        {
            let close = if open == '\u{201c}' { '\u{201d}' } else { open };
            let after = start + open.len_utf8();
            let Some(end) = rest[after..].find(close) else {
                break;
            };
            skeleton.push_str(&rest[..start]);
            skeleton.push(' ');
            rest = &rest[after + end + close.len_utf8()..];
        }
        skeleton.push_str(rest);
        Self(skeleton)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_quoted_literals() {
        assert_eq!(
            Skeleton::of("Show \u{201c}Free shipping\u{201d} and \"Remove\" in `renderCart`.")
                .as_str(),
            "Show   and   in  ."
        );
        assert_eq!(
            Skeleton::of("It says \"unterminated").as_str(),
            "It says \"unterminated"
        );
    }
}
