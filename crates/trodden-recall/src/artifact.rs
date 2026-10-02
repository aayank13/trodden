use crate::Skeleton;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Artifact {
    Field,
    Table,
    Index,
    Constraint,
    Command,
    Flag,
    Function,
    Type,
    Endpoint,
    Ui,
    Message,
    Locale,
    Style,
    Script,
    Config,
}

impl Artifact {
    const WORDS: &[(&str, &[Self])] = &[
        ("column", &[Self::Field]),
        ("columns", &[Self::Field]),
        ("field", &[Self::Field]),
        ("fields", &[Self::Field]),
        ("attribute", &[Self::Field]),
        ("attributes", &[Self::Field]),
        ("table", &[Self::Table]),
        ("tables", &[Self::Table]),
        ("index", &[Self::Index]),
        ("indexes", &[Self::Index]),
        ("indices", &[Self::Index]),
        ("constraint", &[Self::Constraint]),
        ("constraints", &[Self::Constraint]),
        ("command", &[Self::Command]),
        ("commands", &[Self::Command]),
        ("subcommand", &[Self::Command]),
        ("subcommands", &[Self::Command]),
        ("flag", &[Self::Flag, Self::Field]),
        ("flags", &[Self::Flag, Self::Field]),
        ("option", &[Self::Flag]),
        ("options", &[Self::Flag]),
        ("function", &[Self::Function]),
        ("functions", &[Self::Function]),
        ("helper", &[Self::Function]),
        ("helpers", &[Self::Function]),
        ("method", &[Self::Function]),
        ("methods", &[Self::Function]),
        ("enum", &[Self::Type]),
        ("enums", &[Self::Type]),
        ("struct", &[Self::Type]),
        ("structs", &[Self::Type]),
        ("class", &[Self::Type]),
        ("classes", &[Self::Type]),
        ("trait", &[Self::Type]),
        ("traits", &[Self::Type]),
        ("interface", &[Self::Type]),
        ("interfaces", &[Self::Type]),
        ("endpoint", &[Self::Endpoint]),
        ("endpoints", &[Self::Endpoint]),
        ("route", &[Self::Endpoint]),
        ("routes", &[Self::Endpoint]),
        ("ui", &[Self::Ui]),
        ("button", &[Self::Ui]),
        ("buttons", &[Self::Ui]),
        ("link", &[Self::Ui]),
        ("links", &[Self::Ui]),
        ("badge", &[Self::Ui]),
        ("badges", &[Self::Ui]),
        ("heading", &[Self::Ui]),
        ("headings", &[Self::Ui]),
        ("tooltip", &[Self::Ui]),
        ("tooltips", &[Self::Ui]),
        ("banner", &[Self::Ui]),
        ("message", &[Self::Message]),
        ("messages", &[Self::Message]),
        ("label", &[Self::Message]),
        ("labels", &[Self::Message]),
        ("translation", &[Self::Message]),
        ("translations", &[Self::Message]),
        ("translatable", &[Self::Message]),
        ("translated", &[Self::Message]),
        ("locale", &[Self::Locale]),
        ("locales", &[Self::Locale]),
        ("language", &[Self::Locale]),
        ("languages", &[Self::Locale]),
        ("stylesheet", &[Self::Style]),
        ("stylesheets", &[Self::Style]),
        ("css", &[Self::Style]),
        ("theme", &[Self::Style]),
        ("script", &[Self::Script]),
        ("scripts", &[Self::Script]),
        ("config", &[Self::Config]),
        ("configuration", &[Self::Config]),
        ("setting", &[Self::Config]),
        ("settings", &[Self::Config]),
    ];

    fn named_by(word: &str) -> &'static [Self] {
        Self::WORDS
            .iter()
            .find(|(known, _)| *known == word)
            .map_or(&[], |(_, artifacts)| artifacts)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Artifacts(u16);

impl Artifacts {
    pub fn requested(prompt: &str) -> Self {
        Self::named(prompt).into_iter().next().unwrap_or_default()
    }

    pub fn mentioned<'a>(prompts: impl IntoIterator<Item = &'a str>) -> Self {
        prompts
            .into_iter()
            .flat_map(Self::named)
            .fold(Self::default(), |set, named| Self(set.0 | named.0))
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub fn contains(self, artifact: Artifact) -> bool {
        self.0 & Self::bit(artifact) != 0
    }

    fn bit(artifact: Artifact) -> u16 {
        1 << artifact as u8
    }

    fn named(prompt: &str) -> Vec<Self> {
        Skeleton::of(prompt)
            .as_str()
            .to_lowercase()
            .split(|c: char| !(c.is_alphanumeric() || matches!(c, '-' | '_')))
            .map(|word| {
                Self(
                    Artifact::named_by(word)
                        .iter()
                        .fold(0, |bits, &artifact| bits | Self::bit(artifact)),
                )
            })
            .filter(|named| !named.is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_named_thing_is_the_request() {
        let cases = [
            (
                "Add a `--json` flag to `invoicer list`.",
                Some(Artifact::Flag),
            ),
            ("Add a `count` subcommand.", Some(Artifact::Command)),
            (
                "Add a function that sums invoice amounts, for other commands to use.",
                Some(Artifact::Function),
            ),
            (
                "Add an index on tasks.project_id to speed up listing tasks.",
                Some(Artifact::Index),
            ),
            (
                "Add an `index` subcommand that prints invoice numbers.",
                Some(Artifact::Command),
            ),
            ("Give tasks a `priority` that is 0 unless set.", None),
        ];
        for (prompt, artifact) in cases {
            let requested = Artifacts::requested(prompt);
            match artifact {
                Some(artifact) => assert!(requested.contains(artifact), "{prompt}"),
                None => assert!(requested.is_empty(), "{prompt}"),
            }
        }
    }

    #[test]
    fn a_flag_can_be_a_column() {
        let requested = Artifacts::requested("Add a `blocked` flag to tasks.");
        let learned = Artifacts::mentioned(["Add an integer `priority` column to tasks."]);
        assert!(requested.intersects(learned));
        let learned = Artifacts::mentioned(["Add a `count` subcommand."]);
        assert!(!requested.intersects(learned));
    }

    #[test]
    fn procedures_cover_everything_their_prompts_name() {
        let learned = Artifacts::mentioned([
            "Add a \"Proceed to checkout\" button. Its label must be a new translatable message.",
        ]);
        assert!(learned.contains(Artifact::Ui));
        assert!(learned.contains(Artifact::Message));
        assert!(!learned.contains(Artifact::Locale));
        assert!(learned.intersects(Artifacts::requested("Add a translated thank-you line.")));
        assert!(!learned.intersects(Artifacts::requested("Add Spanish as a fourth locale.")));
    }
}
