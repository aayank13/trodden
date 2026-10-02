use jiff::Timestamp;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{HarnessId, ProcedureId, SessionId};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Trace {
    pub session: SessionId,
    pub harness: HarnessId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    pub started_at: Timestamp,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Event {
    pub seq: u32,
    pub at: Timestamp,
    #[serde(flatten)]
    pub kind: EventKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum EventKind {
    Prompt {
        summary: String,
    },
    ToolCall(ToolCall),
    Compaction {
        automatic: bool,
    },
    Subagent {
        session: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_type: Option<String>,
    },
    Injection(Injection),
    SessionEnd,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCall {
    pub tool: String,
    pub action: ToolAction,
    #[serde(default, skip_serializing_if = "ToolArgs::is_empty")]
    pub args: ToolArgs,
    pub outcome: ToolOutcome,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub changes: Vec<FileChange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolAction {
    Read,
    Search,
    Edit,
    Run,
    Fetch,
    Delegate,
    Other,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct ToolArgs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub command: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
}

impl ToolArgs {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum ToolOutcome {
    Succeeded,
    Failed {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        exit_code: Option<i32>,
    },
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct FileChange {
    pub path: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub created: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub symbols: Vec<String>,
    pub lines_added: u32,
    pub lines_removed: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Injection {
    pub procedure: ProcedureId,
    pub revision: u32,
    pub holdout: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_a_failed_command() {
        let json = r#"{
            "session": "0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90",
            "harness": "claude-code",
            "cwd": "~/src/catalog",
            "started_at": "2026-09-21T14:02:11Z",
            "events": [
                {
                    "seq": 0,
                    "at": "2026-09-21T14:02:19Z",
                    "type": "tool_call",
                    "tool": "Bash",
                    "action": "run",
                    "args": { "command": "node --test test/" },
                    "outcome": { "status": "failed", "exit_code": 1 }
                }
            ]
        }"#;
        let trace: Trace = serde_json::from_str(json).expect("valid trace");

        let [event] = trace.events.as_slice() else {
            panic!("one event");
        };
        let EventKind::ToolCall(call) = &event.kind else {
            panic!("a tool call");
        };
        assert_eq!(call.args.command.as_deref(), Some("node --test test/"));
        assert!(matches!(call.outcome, ToolOutcome::Failed { .. }));
    }
}
