use std::{collections::HashMap, path::PathBuf};

use anyhow::{Result, bail};
use jiff::Timestamp;
use serde::Deserialize;
use serde_json::{Map, Value};
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

    const PASTED: &str = "<pasted_content";

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
        for line in text.lines().filter_map(Line::parse) {
            builder.push(line);
        }
        builder.finish()
    }
}

#[derive(Debug)]
struct Line {
    kind: String,
    session_id: Option<String>,
    cwd: Option<String>,
    timestamp: Option<String>,
    is_meta: bool,
    is_sidechain: bool,
    is_compact_summary: bool,
    subtype: Option<String>,
    compact_trigger: Option<String>,
    message: Option<Message>,
    tool_use_result: Option<Value>,
    attachment: Option<Attachment>,
}

impl Line {
    const REPLACEMENT: &str = r"�";

    fn parse(text: &str) -> Option<Self> {
        let value: Value = serde_json::from_str(text)
            .or_else(|_| serde_json::from_str(&Self::without_lone_surrogates(text)))
            .ok()?;
        let mut fields = Fields::of(value)?;
        Some(Self {
            kind: fields.string("type")?,
            session_id: fields.string("sessionId"),
            cwd: fields.string("cwd"),
            timestamp: fields.string("timestamp"),
            is_meta: fields.flag("isMeta"),
            is_sidechain: fields.flag("isSidechain"),
            is_compact_summary: fields.flag("isCompactSummary"),
            subtype: fields.string("subtype"),
            compact_trigger: fields
                .object("compactMetadata")
                .and_then(|mut metadata| metadata.string("trigger")),
            message: fields.object("message").map(Message::new),
            tool_use_result: fields.take("toolUseResult"),
            attachment: fields.object("attachment").and_then(Attachment::new),
        })
    }

    fn without_lone_surrogates(text: &str) -> String {
        let mut mended = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(at) = rest.find('\\') {
            mended.push_str(&rest[..at]);
            rest = &rest[at..];
            let (kept, len) = match Self::code_unit(rest) {
                Some(0xD800..=0xDBFF)
                    if matches!(Self::code_unit(&rest[6..]), Some(0xDC00..=0xDFFF)) =>
                {
                    (&rest[..12], 12)
                }
                Some(0xD800..=0xDFFF) => (Self::REPLACEMENT, 6),
                Some(_) => (&rest[..6], 6),
                None => {
                    let len = rest[1..].chars().next().map_or(1, |c| 1 + c.len_utf8());
                    (&rest[..len], len)
                }
            };
            mended.push_str(kept);
            rest = &rest[len..];
        }
        mended.push_str(rest);
        mended
    }

    fn code_unit(escape: &str) -> Option<u16> {
        let hex = escape
            .strip_prefix(r"\u")?
            .get(..4)
            .filter(|hex| hex.bytes().all(|b| b.is_ascii_hexdigit()))?;
        u16::from_str_radix(hex, 16).ok()
    }
}

#[derive(Debug)]
struct Fields(Map<String, Value>);

impl Fields {
    fn of(value: Value) -> Option<Self> {
        match value {
            Value::Object(map) => Some(Self(map)),
            _ => None,
        }
    }

    fn take(&mut self, key: &str) -> Option<Value> {
        self.0.remove(key)
    }

    fn string(&mut self, key: &str) -> Option<String> {
        match self.take(key)? {
            Value::String(text) => Some(text),
            _ => None,
        }
    }

    fn object(&mut self, key: &str) -> Option<Self> {
        self.take(key).and_then(Self::of)
    }

    fn flag(&self, key: &str) -> bool {
        self.0.get(key).and_then(Value::as_bool) == Some(true)
    }
}

#[derive(Debug)]
struct Message {
    model: Option<String>,
    content: Content,
}

impl Message {
    fn new(mut fields: Fields) -> Self {
        Self {
            model: fields.string("model"),
            content: Content::new(fields.take("content")),
        }
    }
}

#[derive(Debug)]
struct Attachment {
    kind: String,
    mode: Option<String>,
    origin: Option<String>,
    prompt: Content,
}

impl Attachment {
    fn new(mut fields: Fields) -> Option<Self> {
        Some(Self {
            kind: fields.string("type")?,
            mode: fields.string("commandMode"),
            origin: fields
                .object("origin")
                .and_then(|mut origin| origin.string("kind")),
            prompt: Content::new(fields.take("prompt")),
        })
    }

    fn queued_prompt(self) -> Option<String> {
        let typed = self.kind == "queued_command"
            && self.mode.as_deref() == Some("prompt")
            && self.origin.as_deref() == Some("human");
        if !typed {
            return None;
        }
        self.prompt.first_text()
    }
}

#[derive(Debug)]
enum Content {
    Text(String),
    Blocks(Vec<Block>),
}

impl Content {
    fn new(value: Option<Value>) -> Self {
        match value {
            Some(Value::String(text)) => Self::Text(text),
            Some(Value::Array(blocks)) => {
                Self::Blocks(blocks.into_iter().filter_map(Block::new).collect())
            }
            _ => Self::Blocks(Vec::new()),
        }
    }

    fn first_text(self) -> Option<String> {
        match self {
            Self::Text(text) => Some(text),
            Self::Blocks(blocks) => blocks.into_iter().find_map(|block| match block {
                Block::Text { text } => Some(text),
                _ => None,
            }),
        }
    }
}

#[derive(Debug)]
enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: Value,
        is_error: bool,
    },
}

impl Block {
    fn new(value: Value) -> Option<Self> {
        let mut fields = Fields::of(value)?;
        match fields.string("type")?.as_str() {
            "text" => Some(Self::Text {
                text: fields.string("text")?,
            }),
            "tool_use" => Some(Self::ToolUse {
                id: fields.string("id")?,
                name: fields.string("name")?,
                input: fields.take("input").unwrap_or_default(),
            }),
            "tool_result" => Some(Self::ToolResult {
                tool_use_id: fields.string("tool_use_id")?,
                content: fields.take("content").unwrap_or_default(),
                is_error: fields.flag("is_error"),
            }),
            _ => None,
        }
    }
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
            "attachment" => {
                if let Some(text) = line.attachment.and_then(Attachment::queued_prompt) {
                    self.push_prompt(&text);
                }
            }
            "system" if line.subtype.as_deref() == Some("compact_boundary") => {
                let automatic = line
                    .compact_trigger
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
        let headline = Self::headline(text);
        if headline.is_empty() {
            return;
        }
        let summary = self.summary(headline);
        self.emit(EventKind::Prompt { summary });
    }

    fn headline(text: &str) -> &str {
        let mut wrapper: Option<&str> = None;
        let mut pasted = None;
        for line in text.lines() {
            let mut rest = line.trim();
            while !rest.is_empty() {
                if let Some(tag) = wrapper {
                    let Some(close) = rest.find(&format!("</{}", &tag[1..])) else {
                        if tag == Transcript::PASTED {
                            pasted.get_or_insert(rest);
                        }
                        break;
                    };
                    let inside = rest[..close].trim();
                    if tag == Transcript::PASTED && !inside.is_empty() {
                        pasted.get_or_insert(inside);
                    }
                    wrapper = None;
                    rest = Self::after_tag(&rest[close..]);
                } else if let Some(tag) = Self::wrapper(rest) {
                    wrapper = Some(tag);
                    rest = Self::after_tag(rest);
                } else {
                    return rest;
                }
            }
        }
        pasted.unwrap_or_default()
    }

    fn wrapper(line: &str) -> Option<&'static str> {
        Transcript::NON_PROMPT_PREFIXES
            .iter()
            .chain([&Transcript::PASTED])
            .map(|prefix| prefix.trim_end_matches('>'))
            .filter(|tag| tag.starts_with('<'))
            .find(|tag| line.starts_with(tag))
    }

    fn after_tag(text: &str) -> &str {
        text.find('>').map_or("", |end| text[end + 1..].trim())
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

        fn resolved(name: &str, input: Value, result: &str) -> ToolCall {
            let call = json!({"type": "assistant", "sessionId": "s", "cwd": "/work/app",
                              "timestamp": "2026-09-21T14:02:12Z",
                              "message": {"role": "assistant", "content": [
                                  {"type": "tool_use", "id": "call", "name": name, "input": input}]}});
            let trace = Transcript::parse(
                &format!("{call}\n{result}\n"),
                &Redactor::with_home("/home/dev"),
            )
            .expect("valid transcript");
            trace
                .events
                .into_iter()
                .find_map(|event| match event.kind {
                    EventKind::ToolCall(call) => Some(call),
                    _ => None,
                })
                .expect("one tool call")
        }

        fn result(content: &str, is_error: bool, tool_use_result: Value) -> Value {
            json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:13Z",
                   "toolUseResult": tool_use_result,
                   "message": {"role": "user", "content": [
                       {"type": "tool_result", "tool_use_id": "call", "content": content, "is_error": is_error}]}})
        }

        fn prompts(lines: &[Value]) -> Vec<String> {
            let text: String = lines.iter().map(|line| format!("{line}\n")).collect();
            let trace = Transcript::parse(&text, &Redactor::with_home("/home/dev"))
                .expect("valid transcript");
            trace
                .events
                .into_iter()
                .filter_map(|event| match event.kind {
                    EventKind::Prompt { summary } => Some(summary),
                    _ => None,
                })
                .collect()
        }

        fn queued(prompt: Value, mode: &str, origin: Value, sidechain: bool) -> Value {
            json!({"type": "attachment", "sessionId": "s", "cwd": "/work/app",
                   "timestamp": "2026-09-21T14:02:14Z", "isSidechain": sidechain,
                   "attachment": {"type": "queued_command", "prompt": prompt, "commandMode": mode,
                                  "origin": origin, "timestamp": "2026-09-21T14:02:14Z"}})
        }

        fn odd_variants(line: &Value) -> Vec<String> {
            let with_block = |block: Value| {
                let mut line = line.clone();
                line["message"]["content"]
                    .as_array_mut()
                    .expect("content blocks")
                    .insert(0, block);
                line.to_string()
            };
            let mut meta = line.clone();
            meta["isMeta"] = Value::Null;
            vec![
                line.to_string(),
                meta.to_string(),
                with_block(json!({"type": "text"})),
                with_block(json!({"type": "text", "text": null})),
                with_block(json!({"type": "tool_result", "tool_use_id": null, "is_error": "yes"})),
                with_block(json!({"type": "text", "text": "cut mid emoji SURROGATE"}))
                    .replace("SURROGATE", r"\ud83d"),
            ]
        }
    }

    #[test]
    fn pasted_content_does_not_become_the_summary() {
        let cases = [
            "<pasted_content id=\"f3c9\">\nTypeError: total is undefined\n    at invoiceTotal (src/invoice.js:42:17)\n</pasted_content id=\"f3c9\">\n\nFix this crash in the invoice total",
            "\n\n<pasted_content id=\"f3c9\">TypeError: total is undefined</pasted_content id=\"f3c9\"> Fix this crash in the invoice total\n",
            "<pasted_content id=\"a1\">\nTypeError: total is undefined\n</pasted_content id=\"a1\">\n<pasted_content id=\"b2\">\nexpected 42\n</pasted_content id=\"b2\">\nFix this crash in the invoice total",
            "<pasted_content id=\"a1\">\nTypeError: total is undefined\n</pasted_content id=\"a1\">\n<system-reminder>\nThe user opened src/invoice.js\n</system-reminder>\nFix this crash in the invoice total",
            "Fix this crash in the invoice total\n<pasted_content id=\"a1\">\nTypeError: total is undefined\n</pasted_content id=\"a1\">",
        ];
        for prompt in cases {
            assert_eq!(
                summary(prompt),
                "Fix this crash in the invoice total",
                "{prompt}"
            );
        }
    }

    #[test]
    fn prompts_of_only_pasted_content_use_its_first_line() {
        let lead = format!("Deploy the billing worker to staging {}", "x".repeat(148));
        let cases = [
            (
                "<pasted_content id=\"a1\">\n\nRename the shipping helper to computeShippingCost\nand update callers\n</pasted_content id=\"a1\">\n\n".to_owned(),
                "Rename the shipping helper to computeShippingCost".to_owned(),
            ),
            (
                "<pasted_content id=\"a1\">Rename the shipping helper</pasted_content id=\"a1\">"
                    .to_owned(),
                "Rename the shipping helper".to_owned(),
            ),
            (
                format!(
                    "<pasted_content id=\"a1\">\n{lead} {} then report back\n</pasted_content id=\"a1\">",
                    anthropic_key()
                ),
                lead,
            ),
            (
                format!(
                    "<pasted_content id=\"a1\">\nexport KEY={}\n</pasted_content id=\"a1\">",
                    anthropic_key()
                ),
                "export KEY=[REDACTED:llm-api-key]".to_owned(),
            ),
        ];
        for (prompt, expected) in cases {
            assert_eq!(summary(&prompt), expected, "{prompt}");
        }
    }

    #[test]
    fn prompts_with_nothing_of_their_own_are_skipped() {
        let prompts = Session::prompts(&[
            json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:11Z",
                   "message": {"role": "user", "content": "<pasted_content id=\"a1\">\n\n</pasted_content id=\"a1\">\n"}}),
            json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:12Z",
                   "message": {"role": "user", "content": "Fix the paging bug"}}),
        ]);

        assert_eq!(prompts, ["Fix the paging bug"]);
    }

    #[test]
    fn prompts_typed_while_the_agent_works_are_captured() {
        let human = json!({"kind": "human"});
        let notification = json!({"kind": "task-notification", "producer": "session-task"});
        let prompts = Session::prompts(&[
            json!({"type": "user", "sessionId": "s", "cwd": "/work/app", "timestamp": "2026-09-21T14:02:11Z",
                   "message": {"role": "user", "content": "Fix the paging bug"}}),
            Session::queued(
                json!("Also rename the shipping helper"),
                "prompt",
                human.clone(),
                false,
            ),
            Session::queued(
                json!([{"type": "text", "text": "Then update the changelog"}]),
                "prompt",
                human.clone(),
                false,
            ),
            Session::queued(
                json!("<task-notification>\nbuild finished\n</task-notification>"),
                "task-notification",
                notification.clone(),
                false,
            ),
            Session::queued(json!("Summarize the build"), "prompt", notification, false),
            Session::queued(json!("Summarize the build"), "prompt", Value::Null, false),
            Session::queued(json!("Summarize the build"), "prompt", human, true),
        ]);

        assert_eq!(
            prompts,
            [
                "Fix the paging bug",
                "Also rename the shipping helper",
                "Then update the changelog",
            ]
        );
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

    #[test]
    fn odd_fields_keep_an_edit_and_its_changes() {
        let input = json!({"file_path": "/work/app/src/paginate.js",
                           "old_string": "page * 10 + 9", "new_string": "page * 10 + 10"});
        let result = Session::result(
            "The file /work/app/src/paginate.js has been updated.",
            false,
            json!({"filePath": "/work/app/src/paginate.js",
                   "originalFile": "function paginate(items, page) {\n  return items.slice(page * 10, page * 10 + 9);\n}\n",
                   "structuredPatch": [{"oldStart": 1, "lines": [
                       " function paginate(items, page) {",
                       "-  return items.slice(page * 10, page * 10 + 9);",
                       "+  return items.slice(page * 10, page * 10 + 10);",
                       " }"]}]}),
        );
        for line in Session::odd_variants(&result) {
            let call = Session::resolved("Edit", input.clone(), &line);

            assert_eq!(call.outcome, ToolOutcome::Succeeded, "{line}");
            assert_eq!(call.changes.len(), 1, "{line}");
            assert_eq!(call.changes[0].path, "src/paginate.js");
            assert_eq!(
                (call.changes[0].lines_added, call.changes[0].lines_removed),
                (1, 1)
            );
        }
    }

    #[test]
    fn odd_fields_keep_a_failure_and_its_error() {
        let result = Session::result(
            "Exit code 1\nError: Cannot find module 'left-pad'",
            true,
            json!({"stdout": "", "stderr": "Error: Cannot find module 'left-pad'", "interrupted": false}),
        );
        for line in Session::odd_variants(&result) {
            let call = Session::resolved("Bash", json!({"command": "npm test"}), &line);

            assert_eq!(
                call.outcome,
                ToolOutcome::Failed { exit_code: Some(1) },
                "{line}"
            );
            assert!(call.error.is_some(), "{line}");
        }
    }

    #[test]
    fn only_lone_surrogates_are_replaced() {
        let cases = [
            (r#"{"text":"cut \ud83d"}"#, r#"{"text":"cut �"}"#),
            (r#"{"text":"\udE00 tail"}"#, r#"{"text":"� tail"}"#),
            (r#"{"text":"\ud83d😀"}"#, r#"{"text":"�😀"}"#),
            (r#"{"text":"😀 é"}"#, r#"{"text":"😀 é"}"#),
            (r#"{"text":"\\ud83d \" é\"}"#, r#"{"text":"\\ud83d \" é\"}"#),
            (r#"{"text":"\ud83"#, r#"{"text":"\ud83"#),
            (r"\", r"\"),
        ];
        for (text, mended) in cases {
            assert_eq!(Line::without_lone_surrogates(text), mended, "{text}");
        }
    }
}
