use std::{collections::HashMap, path::PathBuf};

use anyhow::{Result, bail};
use jiff::Timestamp;
use serde::Deserialize;
use serde_json::Value;
use trodden_core::{
    HarnessId, SessionId, Trace,
    trace::{Event, EventKind, FileChange, ToolAction, ToolArgs, ToolCall, ToolOutcome},
};
use trodden_redact::Redactor;

use crate::{
    ErrorSignature,
    command::Command,
    evidence::CheckOutput,
    symbols::{Hunk, SymbolFinder},
};

pub const HARNESS: &str = "claude-code";

#[derive(Debug, Clone, Deserialize)]
pub struct HookInput {
    pub session_id: String,
    #[serde(default)]
    pub transcript_path: Option<PathBuf>,
    pub cwd: PathBuf,
    pub hook_event_name: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub is_interrupt: bool,
    #[serde(default)]
    pub stop_hook_active: bool,
}

#[derive(Debug)]
pub struct Transcript;

impl Transcript {
    const SUMMARY_CHARS: usize = 200;

    const NON_PROMPT_PREFIXES: &[&str] = &[
        "<command-name>",
        "<command-message>",
        "<command-args>",
        "<local-command",
        "<bash-input>",
        "<bash-stdout>",
        "<bash-stderr>",
        "<system-reminder>",
        "<task-notification>",
        "<user-memory-input>",
        "[Request interrupted",
        "Caveat:",
    ];

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "TodoWrite",
        "TodoRead",
        "TaskCreate",
        "TaskUpdate",
        "TaskGet",
        "TaskList",
        "TaskOutput",
        "TaskStop",
        "ToolSearch",
        "AskUserQuestion",
        "EnterPlanMode",
        "ExitPlanMode",
        "Skill",
        "SendMessage",
        "ListAgents",
        "ScheduleWakeup",
        "Monitor",
    ];

    pub fn parse(text: &str, redactor: &Redactor) -> Result<Trace> {
        let mut builder = TraceBuilder::new(redactor);
        for line in text.lines() {
            if let Ok(line) = serde_json::from_str::<Line>(line) {
                builder.push(line);
            }
        }
        builder.finish()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Line {
    #[serde(rename = "type")]
    kind: String,
    session_id: Option<String>,
    cwd: Option<String>,
    timestamp: Option<String>,
    #[serde(default)]
    is_meta: bool,
    #[serde(default)]
    is_sidechain: bool,
    #[serde(default)]
    is_compact_summary: bool,
    subtype: Option<String>,
    compact_metadata: Option<CompactMetadata>,
    message: Option<Message>,
    tool_use_result: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct CompactMetadata {
    trigger: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Message {
    model: Option<String>,
    #[serde(default)]
    content: Content,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

impl Default for Content {
    fn default() -> Self {
        Self::Blocks(Vec::new())
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Value,
        #[serde(default)]
        is_error: bool,
    },
    #[serde(other)]
    Other,
}

struct TraceBuilder<'a> {
    redactor: &'a Redactor,
    session: Option<String>,
    cwd: String,
    model: Option<String>,
    started_at: Option<Timestamp>,
    now: Timestamp,
    events: Vec<Event>,
    pending: HashMap<String, usize>,
}

impl<'a> TraceBuilder<'a> {
    fn new(redactor: &'a Redactor) -> Self {
        Self {
            redactor,
            session: None,
            cwd: String::new(),
            model: None,
            started_at: None,
            now: Timestamp::UNIX_EPOCH,
            events: Vec::new(),
            pending: HashMap::new(),
        }
    }

    fn push(&mut self, line: Line) {
        if line.is_sidechain {
            return;
        }
        if self.session.is_none() {
            self.session.clone_from(&line.session_id);
        }
        if self.cwd.is_empty()
            && let Some(cwd) = &line.cwd
        {
            cwd.trim_end_matches('/').clone_into(&mut self.cwd);
        }
        if let Some(at) = line
            .timestamp
            .as_deref()
            .and_then(|t| t.parse::<Timestamp>().ok())
        {
            self.now = at;
            self.started_at.get_or_insert(at);
        }

        match line.kind.as_str() {
            "user" => self.push_user(line),
            "assistant" => self.push_assistant(line),
            "system" if line.subtype.as_deref() == Some("compact_boundary") => {
                let automatic = line
                    .compact_metadata
                    .and_then(|meta| meta.trigger)
                    .is_some_and(|trigger| trigger == "auto");
                self.emit(EventKind::Compaction { automatic });
            }
            _ => {}
        }
    }

    fn push_user(&mut self, line: Line) {
        let Some(message) = line.message else { return };
        match message.content {
            Content::Text(text) => {
                if !line.is_meta && !line.is_compact_summary {
                    self.push_prompt(&text);
                }
            }
            Content::Blocks(blocks) => {
                let mut prompt = None;
                for block in blocks {
                    match block {
                        Block::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            self.resolve(
                                &tool_use_id,
                                &content,
                                is_error,
                                line.tool_use_result.as_ref(),
                            );
                        }
                        Block::Text { text } if prompt.is_none() => prompt = Some(text),
                        _ => {}
                    }
                }
                if let Some(text) = prompt.filter(|_| !line.is_meta && !line.is_compact_summary) {
                    self.push_prompt(&text);
                }
            }
        }
    }

    fn push_prompt(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty()
            || Transcript::NON_PROMPT_PREFIXES
                .iter()
                .any(|prefix| text.starts_with(prefix))
        {
            return;
        }
        let first_line = text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap_or_default();
        let truncated: String = first_line.chars().take(Transcript::SUMMARY_CHARS).collect();
        let summary = self.redactor.redact(&truncated).into_owned();
        self.emit(EventKind::Prompt { summary });
    }

    fn push_assistant(&mut self, line: Line) {
        let Some(message) = line.message else { return };
        if self.model.is_none() {
            self.model = message.model.filter(|model| !model.starts_with('<'));
        }
        let Content::Blocks(blocks) = message.content else {
            return;
        };
        for block in blocks {
            if let Block::ToolUse { id, name, input } = block
                && let Some(call) = self.tool_call(&name, &input)
            {
                let index = self.emit(EventKind::ToolCall(call));
                self.pending.insert(id, index);
            }
        }
    }

    fn tool_call(&self, name: &str, input: &Value) -> Option<ToolCall> {
        if Transcript::BOOKKEEPING_TOOLS.contains(&name) {
            return None;
        }
        let field = |key: &str| input.get(key).and_then(Value::as_str);
        let mut args = ToolArgs::default();
        let action = match name {
            "Bash" => {
                let command = Command::normalize(field("command").unwrap_or_default(), &self.cwd);
                args.command = Some(self.redactor.redact(command.text()).into_owned());
                command.action()
            }
            "Read" | "NotebookRead" => {
                args.path = field("file_path")
                    .or_else(|| field("notebook_path"))
                    .map(|p| self.relative(p));
                ToolAction::Read
            }
            "Edit" | "MultiEdit" | "Write" | "NotebookEdit" => {
                args.path = field("file_path")
                    .or_else(|| field("notebook_path"))
                    .map(|p| self.relative(p));
                ToolAction::Edit
            }
            "Glob" | "Grep" => {
                args.pattern = field("pattern").map(|p| self.redactor.redact(p).into_owned());
                args.path = field("path").map(|p| self.relative(p));
                ToolAction::Search
            }
            "WebFetch" => {
                args.url = field("url").map(|u| self.redactor.redact(u).into_owned());
                ToolAction::Fetch
            }
            "WebSearch" => {
                args.query = field("query").map(|q| self.redactor.redact(q).into_owned());
                ToolAction::Fetch
            }
            "Agent" | "Task" => ToolAction::Delegate,
            _ => ToolAction::Other,
        };
        Some(ToolCall {
            tool: name.to_owned(),
            action,
            args,
            outcome: ToolOutcome::Interrupted,
            changes: Vec::new(),
            duration_ms: None,
            error: None,
        })
    }

    fn resolve(
        &mut self,
        tool_use_id: &str,
        content: &Value,
        is_error: bool,
        result: Option<&Value>,
    ) {
        let Some(index) = self.pending.remove(tool_use_id) else {
            return;
        };
        let changes = if is_error {
            Vec::new()
        } else {
            result
                .and_then(|r| self.file_change(r))
                .into_iter()
                .collect()
        };
        let EventKind::ToolCall(call) = &mut self.events[index].kind else {
            return;
        };

        let interrupted = result
            .and_then(|r| r.get("interrupted"))
            .and_then(Value::as_bool)
            == Some(true);
        let text = Self::text(content);
        let verdict = call
            .args
            .command
            .as_deref()
            .and_then(|command| CheckOutput::verdict(command, &text));
        call.outcome = match (interrupted, is_error, verdict) {
            (true, _, _) => ToolOutcome::Interrupted,
            (false, true, verdict) => match Self::failure(&text) {
                ToolOutcome::Failed { .. } if verdict == Some(true) => ToolOutcome::Succeeded,
                outcome => outcome,
            },
            (false, false, Some(false)) => ToolOutcome::Failed { exit_code: None },
            (false, false, _) => ToolOutcome::Succeeded,
        };
        if matches!(call.outcome, ToolOutcome::Failed { .. }) {
            call.error = ErrorSignature::of(&text, self.redactor);
        }
        call.duration_ms = result
            .and_then(|r| r.get("durationMs"))
            .and_then(Value::as_u64);
        if call.action == ToolAction::Edit {
            call.changes = changes;
        }
    }

    fn text(content: &Value) -> String {
        match content {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    fn failure(text: &str) -> ToolOutcome {
        if text.starts_with("The user doesn't want") || text.contains("user rejected") {
            return ToolOutcome::Interrupted;
        }
        let exit_code = text
            .strip_prefix("Exit code ")
            .and_then(|rest| rest.split_whitespace().next())
            .and_then(|code| code.parse().ok());
        ToolOutcome::Failed { exit_code }
    }

    fn file_change(&self, result: &Value) -> Option<FileChange> {
        let path = result.get("filePath").and_then(Value::as_str)?;
        let original = result.get("originalFile").and_then(Value::as_str);
        let hunks = result
            .get("structuredPatch")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let mut changed = Vec::new();
        let mut starts = Vec::new();
        let (mut added, mut removed) = (0_u32, 0_u32);
        for hunk in &hunks {
            if let Some(start) = hunk.get("oldStart").and_then(Value::as_u64) {
                starts.push(Hunk {
                    old_start: usize::try_from(start).unwrap_or(usize::MAX),
                });
            }
            for line in hunk
                .get("lines")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                if let Some(text) = line.strip_prefix('+') {
                    added += 1;
                    changed.push(text);
                } else if let Some(text) = line.strip_prefix('-') {
                    removed += 1;
                    changed.push(text);
                }
            }
        }

        let created = result.get("type").and_then(Value::as_str) == Some("create")
            || (original.is_none() && result.get("content").is_some());
        let content = result.get("content").and_then(Value::as_str);
        if created && hunks.is_empty() {
            let lines: Vec<&str> = content.unwrap_or_default().lines().collect();
            added = u32::try_from(lines.len()).unwrap_or(u32::MAX);
            changed = lines;
        }

        Some(FileChange {
            path: self.relative(path),
            created,
            symbols: SymbolFinder::find(changed, original, &starts),
            lines_added: added,
            lines_removed: removed,
        })
    }

    fn relative(&self, path: &str) -> String {
        if !self.cwd.is_empty() {
            if path.trim_end_matches('/') == self.cwd {
                return ".".to_owned();
            }
            if let Some(relative) = path
                .strip_prefix(&self.cwd)
                .and_then(|rest| rest.strip_prefix('/'))
            {
                return relative.to_owned();
            }
        }
        self.redactor.redact(path).into_owned()
    }

    fn emit(&mut self, kind: EventKind) -> usize {
        let seq = u32::try_from(self.events.len()).unwrap_or(u32::MAX);
        self.events.push(Event {
            seq,
            at: self.now,
            kind,
        });
        self.events.len() - 1
    }

    fn finish(self) -> Result<Trace> {
        let Some(session) = self.session else {
            bail!("no Claude Code session found in transcript");
        };
        Ok(Trace {
            session: SessionId::new(session),
            harness: HarnessId::new(HARNESS),
            model: self.model,
            cwd: self.redactor.redact(&self.cwd).into_owned(),
            commit: None,
            started_at: self.started_at.unwrap_or(self.now),
            events: self.events,
        })
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn rejects_text_without_a_session() {
        let redactor = Redactor::with_home("/home/dev");

        assert!(Transcript::parse("not a transcript\n", &redactor).is_err());
    }

    fn outcome(command: &str, output: &str, is_error: bool) -> ToolOutcome {
        let lines = [
            json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:11Z",
                   "message": {"role": "user", "content": "Fix the paging bug"}}),
            json!({"type": "assistant", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:12Z",
                   "message": {"role": "assistant", "content": [
                       {"type": "tool_use", "id": "run", "name": "Bash", "input": {"command": command}}]}}),
            json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:13Z",
                   "message": {"role": "user", "content": [
                       {"type": "tool_result", "tool_use_id": "run", "content": output, "is_error": is_error}]}}),
        ];
        let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
        let trace =
            Transcript::parse(&text, &Redactor::with_home("/home/dev")).expect("valid transcript");
        trace
            .events
            .into_iter()
            .find_map(|event| match event.kind {
                EventKind::ToolCall(call) => Some(call.outcome),
                _ => None,
            })
            .expect("one tool call")
    }

    #[test]
    fn piped_checks_are_judged_by_their_own_summary() {
        let failing = "running 3 tests\ntest paging ... FAILED\n\ntest result: FAILED. 2 passed; 1 failed; 0 ignored";
        let passing = "running 3 tests\n\ntest result: ok. 3 passed; 0 failed; 0 ignored";

        assert_eq!(
            outcome("cargo test 2>&1 | tail -20", failing, false),
            ToolOutcome::Failed { exit_code: None }
        );
        assert_eq!(
            outcome("cargo test || true", failing, false),
            ToolOutcome::Failed { exit_code: None }
        );
        assert_eq!(
            outcome("cargo test 2>&1 | grep -i failed", passing, true),
            ToolOutcome::Succeeded
        );
        assert_eq!(
            outcome("cargo test 2>&1 | tail -20", passing, false),
            ToolOutcome::Succeeded
        );
    }

    #[test]
    fn plain_commands_keep_their_exit_code() {
        assert_eq!(
            outcome(
                "cargo test",
                "test result: FAILED. 2 passed; 1 failed",
                false
            ),
            ToolOutcome::Succeeded
        );
        assert_eq!(
            outcome(
                "cargo build && cargo test",
                "Exit code 101\ntest result: ok. 3 passed",
                true
            ),
            ToolOutcome::Failed {
                exit_code: Some(101)
            }
        );
        assert_eq!(
            outcome("npm run build | tee build.log", "built in 2.1s", false),
            ToolOutcome::Succeeded
        );
    }

    #[test]
    fn declined_tool_calls_are_interruptions() {
        let content =
            Value::String("The user doesn't want to proceed with this tool use.".to_owned());

        assert_eq!(
            TraceBuilder::failure(&TraceBuilder::text(&content)),
            ToolOutcome::Interrupted
        );
    }
}
