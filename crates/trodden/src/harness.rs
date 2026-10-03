use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use clap::ValueEnum;
use serde::Deserialize;
use trodden_capture::claude_code::{self, Transcript};
use trodden_core::Trace;
use trodden_redact::Redactor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum Harness {
    #[value(name = claude_code::HARNESS)]
    ClaudeCode,
}

impl Harness {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ClaudeCode => claude_code::HARNESS,
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::value_variants()
            .iter()
            .copied()
            .find(|harness| harness.as_str() == name)
    }

    pub fn history_dir(self) -> Result<PathBuf> {
        match self {
            Self::ClaudeCode => ClaudeCode::projects(),
        }
    }

    pub(crate) fn transcripts(self, history: &Path) -> Result<Vec<PathBuf>> {
        match self {
            Self::ClaudeCode => ClaudeCode::transcripts(history),
        }
    }

    pub(crate) fn parse(self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        match self {
            Self::ClaudeCode => Ok((
                Transcript::parse(text, redactor)?,
                ClaudeCode::working_directory(text),
            )),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeCodeLine {
    cwd: Option<PathBuf>,
    #[serde(default)]
    is_sidechain: bool,
}

#[derive(Debug)]
struct ClaudeCode;

impl ClaudeCode {
    fn working_directory(text: &str) -> Option<PathBuf> {
        text.lines()
            .filter_map(|line| serde_json::from_str::<ClaudeCodeLine>(line).ok())
            .filter(|line| !line.is_sidechain)
            .find_map(|line| line.cwd)
    }

    fn projects() -> Result<PathBuf> {
        Ok(
            PathBuf::from(env::var_os("HOME").context("find the home directory")?)
                .join(".claude/projects"),
        )
    }

    fn transcripts(projects: &Path) -> Result<Vec<PathBuf>> {
        let mut transcripts = Vec::new();
        for project in
            fs::read_dir(projects).with_context(|| format!("list {}", projects.display()))?
        {
            let project = project.context("read a project directory entry")?.path();
            let Ok(entries) = fs::read_dir(&project) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
                {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harness_names_round_trip() {
        for harness in Harness::value_variants() {
            assert_eq!(Harness::from_name(harness.as_str()), Some(*harness));
        }
        assert_eq!(Harness::ClaudeCode.as_str(), "claude-code");
        assert_eq!(Harness::from_name("future-agent"), None);
        assert_eq!(Harness::from_name("Claude-Code"), None);
    }
}
