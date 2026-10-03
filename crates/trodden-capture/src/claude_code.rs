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

    const SUMMARY_SOURCE_BYTES: usize = 64 * 1024;

    const REDACTED: &str = "[REDACTED:";

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
    here: String,
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
            here: String::new(),
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
        if let Some(cwd) = &line.cwd {
            let cwd = cwd.trim_end_matches('/');
            if self.cwd.is_empty() {
                cwd.clone_into(&mut self.cwd);
            }
            cwd.clone_into(&mut self.here);
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
        let summary = self.summary(first_line);
        self.emit(EventKind::Prompt { summary });
    }

    fn summary(&self, line: &str) -> String {
        let source = if line.len() > Transcript::SUMMARY_SOURCE_BYTES {
            line[..line.floor_char_boundary(Transcript::SUMMARY_SOURCE_BYTES)]
                .trim_end_matches(|c: char| !c.is_whitespace())
        } else {
            line
        };
        let redacted = self.redactor.redact(source);
        Self::truncate(&redacted, Transcript::SUMMARY_CHARS)
            .trim_end()
            .to_owned()
    }

    fn truncate(text: &str, chars: usize) -> &str {
        let Some((cut, _)) = text.char_indices().nth(chars) else {
            return text;
        };
        let straddling = text
            .match_indices(Transcript::REDACTED)
            .map(|(start, _)| start)
            .take_while(|&start| start < cut)
            .last()
            .filter(|&start| text[start..].find(']').is_none_or(|end| start + end >= cut));
        &text[..straddling.unwrap_or(cut)]
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
                let subdirectory = self.subdirectory();
                let base = if subdirectory.is_some() {
                    &self.here
                } else {
                    &self.cwd
                };
                let command = Command::normalize(field("command").unwrap_or_default(), base);
                let text = match subdirectory {
                    Some(dir) if !command.text().is_empty() => {
                        format!("cd {} && {}", Self::quoted(dir), command.text())
                    }
                    _ => command.text().to_owned(),
                };
                args.command = Some(self.redactor.redact(&text).into_owned());
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
        match self.subdirectory() {
            Some(dir) if !path.starts_with(['/', '~']) => {
                self.redactor.redact(&Self::joined(dir, path)).into_owned()
            }
            _ => self.redactor.redact(path).into_owned(),
        }
    }

    fn subdirectory(&self) -> Option<&str> {
        if self.cwd.is_empty() {
            return None;
        }
        self.here
            .strip_prefix(&self.cwd)?
            .strip_prefix('/')
            .filter(|dir| !dir.is_empty())
    }

    fn joined(dir: &str, path: &str) -> String {
        let mut parts: Vec<&str> = dir.split('/').collect();
        for part in path.split('/') {
            match part {
                "" | "." => {}
                ".." if parts.last().is_some_and(|last| *last != "..") => {
                    parts.pop();
                }
                _ => parts.push(part),
            }
        }
        if parts.is_empty() {
            ".".to_owned()
        } else {
            parts.join("/")
        }
    }

    fn quoted(dir: &str) -> String {
        if dir
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-/+@%,:=".contains(c))
        {
            dir.to_owned()
        } else {
            format!("'{}'", dir.replace('\'', r"'\''"))
        }
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

    fn summary(prompt: &str) -> String {
        let line = json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:11Z",
                          "message": {"role": "user", "content": prompt}});
        let trace = Transcript::parse(&format!("{line}\n"), &Redactor::with_home("/home/dev"))
            .expect("valid transcript");
        trace
            .events
            .into_iter()
            .find_map(|event| match event.kind {
                EventKind::Prompt { summary } => Some(summary),
                _ => None,
            })
            .expect("one prompt")
    }

    fn anthropic_key() -> String {
        ["sk", "ant", "api03", "Ab3Zq8Lm3KpQ7vX2nB9wR4tY6uI1oP5aS0dF"].join("-")
    }

    #[test]
    fn secrets_across_the_summary_cut_leave_no_fragment() {
        let lead = format!("Deploy the billing worker to staging {}", "x".repeat(148));
        let prompt = format!(
            "{lead} {} then report back\nand nothing else",
            anthropic_key()
        );

        assert_eq!(summary(&prompt), lead);
    }

    #[test]
    fn summaries_keep_whole_markers_or_none() {
        let secrets = [
            anthropic_key(),
            ["AKIA", "IOSFODNN7EXAMPLE"].concat(),
            ["ghp", "R4tY6uI1oP5aS0dFAb3Zq8Lm3KpQ7vX2nB9w"].join("_"),
            "API_KEY=4f9a1c2e8b7d6a5f3e2c1b0a9d8e7f6c".to_owned(),
            "Q7vX2nB9wR4tY6uI1oP5aS0dFAb3Zq8Lm3Kp".to_owned(),
        ];
        let redactor = Redactor::with_home("/home/dev");
        for secret in &secrets {
            for lead in 150..=Transcript::SUMMARY_CHARS {
                let prompt = format!("{} {secret} to deploy staging", "x".repeat(lead));
                let redacted = redactor.redact(&prompt);
                let summary = summary(&prompt);
                let markers = summary.matches(Transcript::REDACTED).count();

                assert!(redacted.contains(Transcript::REDACTED), "{secret}");
                assert!(summary.chars().count() <= Transcript::SUMMARY_CHARS);
                assert!(redacted.starts_with(&summary), "{summary}");
                assert_eq!(summary.matches('[').count(), markers, "{summary}");
                assert_eq!(summary.matches(']').count(), markers, "{summary}");
            }
        }
    }

    #[test]
    fn huge_first_lines_drop_the_token_cut_by_the_bound() {
        let value = "a".repeat(Transcript::SUMMARY_SOURCE_BYTES - 41);
        let prompt = format!(
            "Deploy with TOKEN={value} and use {} for the smoke test",
            anthropic_key()
        );

        assert_eq!(
            summary(&prompt),
            "Deploy with TOKEN=[REDACTED:assignment] and use"
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

    #[derive(Debug)]
    struct Session;

    impl Session {
        fn calls(steps: &[(&str, &str, Value)]) -> (Trace, Vec<ToolCall>) {
            let text: String = steps
                .iter()
                .enumerate()
                .map(|(index, (cwd, name, input))| {
                    let line = json!({"type": "assistant", "sessionId": "s", "cwd": cwd,
                                      "timestamp": "2026-09-21T14:02:12Z",
                                      "message": {"role": "assistant", "content": [
                                          {"type": "tool_use", "id": index.to_string(), "name": name, "input": input}]}});
                    format!("{line}\n")
                })
                .collect();
            let trace = Transcript::parse(&text, &Redactor::with_home("/home/dev"))
                .expect("valid transcript");
            let calls = trace
                .events
                .iter()
                .filter_map(|event| match &event.kind {
                    EventKind::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect();
            (trace, calls)
        }
    }

    #[test]
    fn commands_run_in_a_subdirectory_change_into_it_first() {
        let (trace, calls) = Session::calls(&[
            ("/work/app", "Bash", json!({"command": "cargo test"})),
            ("/work/app/web", "Bash", json!({"command": "npm test"})),
            (
                "/work/app/web/",
                "Bash",
                json!({"command": "cat /work/app/web/src/paginate.js"}),
            ),
            (
                "/work/app/web",
                "Bash",
                json!({"command": "cd /work/app/web && npm run lint"}),
            ),
            ("/work/app/my web", "Bash", json!({"command": "npm test"})),
            (
                "/work/app",
                "Bash",
                json!({"command": "cd web && npm test"}),
            ),
            ("/work/other", "Bash", json!({"command": "npm test"})),
            ("/work/application", "Bash", json!({"command": "npm test"})),
        ]);
        let commands: Vec<&str> = calls
            .iter()
            .map(|call| call.args.command.as_deref().expect("bash command"))
            .collect();

        assert_eq!(trace.cwd, "/work/app");
        assert_eq!(
            commands,
            [
                "cargo test",
                "cd web && npm test",
                "cd web && cat ./src/paginate.js",
                "cd web && npm run lint",
                "cd 'my web' && npm test",
                "cd web && npm test",
                "npm test",
                "npm test",
            ]
        );
        assert_eq!(calls[3].action, ToolAction::Run);
        assert_eq!(calls[2].action, ToolAction::Read);
    }

    #[test]
    fn paths_from_a_subdirectory_are_relative_to_the_trace_root() {
        let (_, calls) = Session::calls(&[
            (
                "/work/app",
                "Grep",
                json!({"pattern": "paginate", "path": "src"}),
            ),
            (
                "/work/app/web",
                "Grep",
                json!({"pattern": "paginate", "path": "src"}),
            ),
            (
                "/work/app/web",
                "Glob",
                json!({"pattern": "*.md", "path": "../docs"}),
            ),
            (
                "/work/app/web",
                "Glob",
                json!({"pattern": "*.md", "path": "."}),
            ),
            (
                "/work/app/web",
                "Glob",
                json!({"pattern": "*.md", "path": "../../shared"}),
            ),
            (
                "/work/app/web",
                "Edit",
                json!({"file_path": "/work/app/web/src/paginate.js"}),
            ),
            (
                "/work/other",
                "Grep",
                json!({"pattern": "paginate", "path": "src"}),
            ),
        ]);
        let paths: Vec<&str> = calls
            .iter()
            .map(|call| call.args.path.as_deref().expect("path argument"))
            .collect();

        assert_eq!(
            paths,
            [
                "src",
                "web/src",
                "docs",
                "web",
                "../shared",
                "web/src/paginate.js",
                "src",
            ]
        );
    }
}
