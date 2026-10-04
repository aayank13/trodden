use std::iter;

use jiff::Timestamp;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{FamilyId, HarnessId, ProcedureId, RepoId, SessionId, TaskKind};

mod anti_unify;
#[cfg(any(test, feature = "test-support"))]
mod example;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Procedure {
    pub id: ProcedureId,
    pub family: FamilyId,
    pub revision: u32,
    pub title: String,
    pub scope: Scope,
    pub trigger: Trigger,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub preconditions: Vec<Condition>,
    pub steps: Vec<Step>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<Verification>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub avoid: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub slots: Vec<Slot>,
    pub provenance: Provenance,
    pub outcomes: Outcomes,
    pub state: Lifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Scope {
    Global,
    Repo { repo: RepoId },
    Paths { repo: RepoId, globs: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Trigger {
    pub entities: Vec<Entity>,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
}

impl Trigger {
    pub fn kind(&self) -> Option<TaskKind> {
        iter::once(&self.text)
            .chain(&self.examples)
            .find_map(|prompt| TaskKind::of(prompt))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Entity {
    Path(String),
    Symbol(String),
    Command(String),
    ErrorSignature(String),
    Package(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Condition {
    FileExists { path: String },
    SymbolInFile { path: String, symbol: String },
    ProgramOnPath { program: String },
    EnvVarSet { name: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Step {
    pub kind: StepKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<Resource>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<Resource>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StepKind {
    Read,
    Search,
    Edit,
    Create,
    Run,
    Verify,
    Setup,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
#[non_exhaustive]
pub enum Resource {
    File(String),
    EnvVar(String),
    Package(String),
    Port(u16),
    Service(String),
    Artifact(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Verification {
    pub command: String,
    pub expect_exit: i32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Slot {
    pub name: String,
    pub kind: SlotKind,
    pub examples: Vec<String>,
}

impl Slot {
    pub const MAX_EXAMPLES: usize = 5;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SlotKind {
    Path,
    Identifier,
    Package,
    Text,
    Number,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Provenance {
    pub sources: Vec<Source>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Source {
    pub session: SessionId,
    pub harness: HarnessId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Outcomes {
    pub successes: u32,
    pub failures: u32,
    pub injections: u32,
    pub holdouts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_success_at: Option<Timestamp>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Lifecycle {
    #[default]
    Candidate,
    Active,
    Stale,
    Archived,
    Quarantined,
    Retired,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_trigger_takes_the_kind_of_its_first_clear_prompt() {
        let trigger = |text: &str, examples: &[&str]| Trigger {
            entities: Vec::new(),
            text: text.to_owned(),
            examples: examples
                .iter()
                .map(|example| (*example).to_owned())
                .collect(),
        };

        assert_eq!(
            trigger("Page 2 repeats the last product. Fix it.", &["Add sorting"]).kind(),
            Some(TaskKind::Fix)
        );
        assert_eq!(
            trigger(
                "perPage values above 100 should be capped",
                &["Add sorting"]
            )
            .kind(),
            Some(TaskKind::Add)
        );
        assert_eq!(
            trigger("perPage values above 100 should be capped", &[]).kind(),
            None
        );
    }
}
