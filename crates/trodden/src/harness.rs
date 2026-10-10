mod claude_code;
mod cline;
mod codex;
mod copilot;
mod cursor;
mod droid;
mod gemini;
mod kimi;
mod opencode;
mod qwen;

use std::{
    fmt::Debug,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::ValueEnum;
use trodden_capture::journal::Observation;
use trodden_core::Trace;
use trodden_redact::Redactor;

use crate::connect::{Change, Program};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, ValueEnum)]
pub enum Harness {
    #[value(name = "claude-code")]
    ClaudeCode,
    #[value(name = "codex")]
    Codex,
    #[value(name = "gemini")]
    Gemini,
    #[value(name = "qwen")]
    Qwen,
    #[value(name = "copilot")]
    Copilot,
    #[value(name = "droid")]
    Droid,
    #[value(name = "cursor")]
    Cursor,
    #[value(name = "opencode")]
    OpenCode,
    #[value(name = "kilo")]
    Kilo,
    #[value(name = "kimi")]
    Kimi,
    #[value(name = "cline")]
    Cline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookEvent {
    pub name: String,
    pub session: String,
    pub cwd: PathBuf,
    pub transcript: Option<PathBuf>,
    pub moment: Moment,
    pub observed: Vec<Observation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Moment {
    SessionStart,
    Prompt(String),
    CommandFailed(String),
    ToolDone,
    TurnEnd { continued: bool },
    Compacting,
    SessionEnd,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reply<'a> {
    Recall(&'a str),
    Remind(&'a str),
}

pub(crate) trait Agent: Debug + Sync {
    fn title(&self) -> &'static str;

    fn journaled(&self) -> bool {
        false
    }

    fn history(&self) -> Result<Option<PathBuf>>;

    fn transcripts(&self, history: &Path) -> Result<Vec<PathBuf>>;

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)>;

    fn event(&self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>>;

    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String>;

    fn detected(&self) -> bool;

    fn connect(&self, program: &Program) -> Result<Vec<Change>>;

    fn notice(&self) -> Option<&'static str> {
        None
    }

    fn disconnect(&self) -> Result<Vec<Change>>;

    fn connected(&self) -> Result<bool>;
}

impl Harness {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
            Self::Gemini => "gemini",
            Self::Qwen => "qwen",
            Self::Copilot => "copilot",
            Self::Droid => "droid",
            Self::Cursor => "cursor",
            Self::OpenCode => "opencode",
            Self::Kilo => "kilo",
            Self::Kimi => "kimi",
            Self::Cline => "cline",
        }
    }

    pub fn all() -> &'static [Self] {
        Self::value_variants()
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::all()
            .iter()
            .copied()
            .find(|harness| harness.as_str() == name)
    }

    fn agent(self) -> &'static dyn Agent {
        match self {
            Self::ClaudeCode => &claude_code::ClaudeCode,
            Self::Codex => &codex::Codex,
            Self::Gemini => &gemini::Gemini,
            Self::Qwen => &qwen::Qwen,
            Self::Copilot => &copilot::Copilot,
            Self::Droid => &droid::Droid,
            Self::Cursor => &cursor::Cursor,
            Self::OpenCode => &opencode::OpenCode::OPENCODE,
            Self::Kilo => &opencode::OpenCode::KILO,
            Self::Kimi => &kimi::Kimi,
            Self::Cline => &cline::Cline,
        }
    }

    pub fn title(self) -> &'static str {
        self.agent().title()
    }

    pub fn journaled(self) -> bool {
        self.agent().journaled()
    }

    pub fn history_dir(self) -> Result<Option<PathBuf>> {
        self.agent().history()
    }

    pub(crate) fn transcripts(self, history: &Path) -> Result<Vec<PathBuf>> {
        self.agent().transcripts(history)
    }

    pub fn read(transcript: &Path) -> Result<String> {
        let bytes = fs::read(transcript).context("read the transcript")?;
        Ok(Self::decode(bytes))
    }

    fn decode(bytes: Vec<u8>) -> String {
        String::from_utf8(bytes)
            .unwrap_or_else(|invalid| String::from_utf8_lossy(invalid.as_bytes()).into_owned())
    }

    pub fn parse(self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        self.agent().parse(text, redactor)
    }

    pub fn event(self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>> {
        self.agent().event(payload, redactor)
    }

    pub fn render(self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        self.agent().render(event, reply)
    }

    pub fn detected(self) -> bool {
        self.agent().detected()
    }

    pub fn connect(self, program: &Program) -> Result<Vec<Change>> {
        self.agent().connect(program)
    }

    pub fn notice(self) -> Option<&'static str> {
        self.agent().notice()
    }

    pub fn disconnect(self) -> Result<Vec<Change>> {
        self.agent().disconnect()
    }

    pub fn connected(self) -> Result<bool> {
        self.agent().connected()
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    #[test]
    fn harness_names_round_trip() {
        for harness in Harness::all() {
            assert_eq!(Harness::from_name(harness.as_str()), Some(*harness));
            assert_eq!(
                harness.to_possible_value().expect("visible").get_name(),
                harness.as_str()
            );
        }
        assert_eq!(Harness::ClaudeCode.as_str(), "claude-code");
        assert_eq!(Harness::from_name("future-agent"), None);
        assert_eq!(Harness::from_name("Claude-Code"), None);
    }

    #[test]
    fn read_errors_leave_the_path_to_the_caller() {
        let missing = env::temp_dir().join(format!(
            "trodden-harness-missing-{}.jsonl",
            std::process::id()
        ));
        let error = Harness::read(&missing).expect_err("a missing transcript fails to read");
        let message = format!("{error:#}");
        assert!(message.starts_with("read the transcript: "), "{message}");
        assert!(!message.contains(&*missing.to_string_lossy()), "{message}");
    }

    #[test]
    fn valid_transcripts_decode_unchanged() {
        let text = "{\"cwd\":\"/home/dev/caf\u{e9}\"}\n";
        assert_eq!(Harness::decode(text.as_bytes().to_vec()), text);
    }
}
