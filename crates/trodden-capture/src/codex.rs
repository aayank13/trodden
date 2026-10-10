use std::{collections::HashMap, path::PathBuf};

use anyhow::{Result, bail};
use serde_json::Value;
use trodden_core::{
    Trace,
    trace::{FileChange, ToolAction, ToolArgs},
};
use trodden_redact::Redactor;

use crate::{
    ErrorSignature,
    builder::{Finish, Prompts, TraceBuilder},
    command::Command,
    diff::Diff,
};

pub const HARNESS: &str = "codex";

#[derive(Debug)]
pub struct Rollout;

impl Rollout {
    const PROMPTS: Prompts = Prompts {
        skipped: &[
            "<environment_context>",
            "<user_instructions>",
            "<user_shell_command>",
            "<turn_aborted>",
            "<subagent_notification>",
            "<hook_prompt",
            "<skill>",
            "# AGENTS.md instructions",
        ],
        pasted: None,
    };

    const SHELL_TOOLS: &[&str] = &["exec_command", "shell_command", "shell", "local_shell"];

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "update_plan",
        "request_user_input",
        "request_permissions",
        "wait",
        "wait_agent",
        "send_input",
        "resume_agent",
        "close_agent",
        "list_agents",
        "create_goal",
        "get_goal",
        "update_goal",
        "tool_search",
        "list_available_plugins_to_install",
        "request_plugin_install",
    ];

    pub fn parse(text: &str, redactor: &Redactor) -> Result<Trace> {
        let lines: Vec<Line> = text.lines().filter_map(Line::parse).collect();
        if lines
            .iter()
            .find(|line| line.kind == "session_meta")
            .is_some_and(|meta| Meta::is_subagent(&meta.payload))
        {
            bail!("Codex subagent rollouts are learned through their parent session");
        }
        let lines = Self::effective(lines);
        let mut reader = Reader::new(redactor, Self::has_user_messages(&lines));
        for line in lines {
            reader.push(line);
        }
        reader.trace()
    }

    pub fn working_directory(text: &str) -> Option<PathBuf> {
        text.lines()
            .filter_map(Line::parse)
            .filter(|line| line.kind == "session_meta" || line.kind == "turn_context")
            .find_map(|line| line.payload.get("cwd")?.as_str().map(PathBuf::from))
    }

    pub fn is_subagent(first_line: &str) -> bool {
        Line::parse(first_line)
            .is_some_and(|line| line.kind == "session_meta" && Meta::is_subagent(&line.payload))
    }

    pub fn failed(command: &str, output: &str, redactor: &Redactor) -> bool {
        if let Some(code) = Exec::parse(output).and_then(|exec| exec.exit_code) {
            return code != 0;
        }
        Command::normalize(command, "").action() == ToolAction::Run
            && ErrorSignature::of(output, redactor).is_some()
    }

    fn has_user_messages(lines: &[Line]) -> bool {
        lines
            .iter()
            .any(|line| line.kind == "event_msg" && line.subtype() == Some("user_message"))
    }

    // A fork copies its parent's history first: turns older than the fork's own id belong to
    // the parent.
    fn effective(lines: Vec<Line>) -> Vec<Line> {
        let mut session: Option<String> = None;
        let mut forked = false;
        let mut copied = false;
        let mut kept: Vec<Line> = Vec::with_capacity(lines.len());
        let mut turns: Vec<usize> = Vec::new();
        for line in lines {
            if line.kind == "session_meta" {
                let id = line.payload.get("id").and_then(Value::as_str);
                match (&session, id) {
                    (None, Some(id)) => session = Some(id.to_owned()),
                    (Some(own), Some(id)) if own != id => {
                        forked = true;
                        copied = true;
                        continue;
                    }
                    (None, None) => {}
                    _ => continue,
                }
            }
            if forked
                && let Some(own) = &session
                && let Some(turn) = line.turn_id()
            {
                copied = turn < own.as_str();
            }
            if copied {
                continue;
            }
            match (line.kind.as_str(), line.subtype()) {
                ("event_msg", Some("user_message")) => turns.push(kept.len()),
                ("event_msg", Some("thread_rolled_back")) => {
                    let count = line
                        .payload
                        .get("num_turns")
                        .and_then(Value::as_u64)
                        .and_then(|count| usize::try_from(count).ok())
                        .unwrap_or_default();
                    if count > 0 && !turns.is_empty() {
                        let first = turns.len().saturating_sub(count);
                        kept.truncate(turns[first]);
                        turns.truncate(first);
                    }
                    continue;
                }
                _ => {}
            }
            kept.push(line);
        }
        kept
    }
}

#[derive(Debug)]
struct Line {
    timestamp: Option<String>,
    kind: String,
    payload: Value,
}

impl Line {
    fn parse(text: &str) -> Option<Self> {
        let Value::Object(mut fields) = serde_json::from_str(text).ok()? else {
            return None;
        };
        let Some(Value::String(kind)) = fields.remove("type") else {
            return None;
        };
        let timestamp = match fields.remove("timestamp") {
            Some(Value::String(at)) => Some(at),
            _ => None,
        };
        Some(Self {
            timestamp,
            kind,
            payload: fields.remove("payload").unwrap_or_default(),
        })
    }

    fn subtype(&self) -> Option<&str> {
        self.payload.get("type").and_then(Value::as_str)
    }

    fn turn_id(&self) -> Option<&str> {
        let starts_turn = self.kind == "turn_context"
            || (self.kind == "event_msg" && self.subtype() == Some("task_started"));
        starts_turn
            .then(|| self.payload.get("turn_id").and_then(Value::as_str))
            .flatten()
    }

    fn field<'v>(value: &'v Value, key: &str) -> Option<&'v str> {
        value.get(key).and_then(Value::as_str)
    }
}

#[derive(Debug)]
struct Meta;

impl Meta {
    fn is_subagent(payload: &Value) -> bool {
        let source = payload.get("source");
        let spawned = source.is_some_and(|source| {
            source.get("subagent").is_some() || source.get("internal").is_some()
        });
        let thread = Line::field(payload, "thread_source")
            .is_some_and(|thread| thread == "subagent" || thread == "memory_consolidation");
        spawned || thread
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Exec<'a> {
    exit_code: Option<i32>,
    running: Option<&'a str>,
    wall_ms: Option<u64>,
    output: &'a str,
}

impl<'a> Exec<'a> {
    fn parse(text: &'a str) -> Option<Self> {
        let mut exec = Self {
            exit_code: None,
            running: None,
            wall_ms: None,
            output: "",
        };
        let mut header = false;
        let mut offset = 0;
        for line in text.split_inclusive('\n') {
            offset += line.len();
            let line = line.trim_end();
            if line == "Output:" {
                exec.output = &text[offset..];
                break;
            }
            if let Some(code) = line
                .strip_prefix("Process exited with code ")
                .or_else(|| line.strip_prefix("Exit code: "))
            {
                exec.exit_code = code.trim().parse().ok();
            } else if let Some(process) = line.strip_prefix("Process running with session ID ") {
                exec.running = Some(process.trim());
            } else if let Some(wall) = line.strip_prefix("Wall time: ") {
                exec.wall_ms = Self::milliseconds(wall);
            } else if ![
                "Chunk ID: ",
                "Original token count: ",
                "Total output lines: ",
            ]
            .iter()
            .any(|prefix| line.starts_with(prefix))
            {
                break;
            }
            header = true;
        }
        header.then_some(exec)
    }

    fn milliseconds(wall: &str) -> Option<u64> {
        let seconds: f64 = wall.trim().strip_suffix("seconds")?.trim().parse().ok()?;
        (seconds.is_finite() && seconds >= 0.0).then(|| {
            #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let ms = (seconds * 1000.0).round() as u64;
            ms
        })
    }

    fn legacy(text: &str) -> Option<(i32, String, Option<u64>)> {
        let value: Value = serde_json::from_str(text.trim_start()).ok()?;
        let metadata = value.get("metadata")?;
        let code = i32::try_from(metadata.get("exit_code")?.as_i64()?).ok()?;
        let wall = metadata
            .get("duration_seconds")
            .and_then(Value::as_f64)
            .and_then(|seconds| Self::milliseconds(&format!("{seconds} seconds")));
        let output = Line::field(&value, "output").unwrap_or_default().to_owned();
        Some((code, output, wall))
    }

    fn interrupted(text: &str) -> bool {
        text.contains("aborted by user") || text.contains("rejected by user")
    }
}

#[derive(Debug)]
struct Running {
    call: String,
    output: String,
    wall_ms: u64,
}

#[derive(Debug)]
struct PatchEnd {
    changes: Vec<FileChange>,
    success: bool,
    declined: bool,
}

struct Reader<'a> {
    builder: TraceBuilder<'a>,
    prompts_from_events: bool,
    cwd: String,
    running: HashMap<String, Running>,
    polls: HashMap<String, String>,
    patched: HashMap<String, PatchEnd>,
    patches: HashMap<String, Vec<FileChange>>,
    turn_prompted: bool,
    compacted: bool,
    searches: usize,
}

impl<'a> Reader<'a> {
    fn new(redactor: &'a Redactor, prompts_from_events: bool) -> Self {
        Self {
            builder: TraceBuilder::new(HARNESS, redactor),
            prompts_from_events,
            cwd: String::new(),
            running: HashMap::new(),
            polls: HashMap::new(),
            patched: HashMap::new(),
            patches: HashMap::new(),
            turn_prompted: false,
            compacted: false,
            searches: 0,
        }
    }

    fn push(&mut self, line: Line) {
        if let Some(at) = &line.timestamp {
            self.builder.time(at);
        }
        let payload = &line.payload;
        match (line.kind.as_str(), line.subtype()) {
            ("session_meta", _) => {
                if let Some(id) = Line::field(payload, "id") {
                    self.builder.session(id);
                }
                self.directory(Line::field(payload, "cwd"));
            }
            ("turn_context", _) => {
                self.directory(Line::field(payload, "cwd"));
                if let Some(model) = Line::field(payload, "model") {
                    self.builder.model(model);
                }
            }
            ("compacted", _) | ("event_msg", Some("context_compacted")) => self.compaction(),
            ("event_msg", Some("task_started")) => {
                self.turn_prompted = false;
                self.compacted = false;
            }
            ("event_msg", Some("user_message")) => {
                if let Some(text) = Line::field(payload, "message") {
                    self.prompt(text);
                }
            }
            ("event_msg", Some("patch_apply_end")) => self.patch_end(payload),
            ("response_item", Some("message")) if !self.prompts_from_events => {
                self.message(payload);
            }
            ("response_item", Some("function_call")) => self.function_call(payload),
            ("response_item", Some("custom_tool_call")) => self.custom_tool_call(payload),
            ("response_item", Some("local_shell_call")) => self.local_shell_call(payload),
            ("response_item", Some("web_search_call")) => self.web_search(payload),
            ("response_item", Some("function_call_output" | "custom_tool_call_output")) => {
                if let Some(id) = Line::field(payload, "call_id") {
                    let text = Self::text(payload.get("output"));
                    self.output(id, &text);
                }
            }
            _ => {}
        }
    }

    fn directory(&mut self, cwd: Option<&str>) {
        if let Some(cwd) = cwd.filter(|cwd| !cwd.is_empty()) {
            self.builder.directory(cwd);
            cwd.clone_into(&mut self.cwd);
        }
    }

    fn prompt(&mut self, text: &str) {
        self.builder.prompt(text, &Rollout::PROMPTS);
        self.turn_prompted = true;
        self.compacted = false;
    }

    fn message(&mut self, payload: &Value) {
        if Line::field(payload, "role") != Some("user") {
            return;
        }
        let text = Self::text(payload.get("content"));
        self.prompt(&text);
    }

    fn compaction(&mut self) {
        if !self.compacted {
            self.builder.compaction(self.turn_prompted);
            self.compacted = true;
        }
    }

    fn call(&mut self, id: &str, tool: &str, action: ToolAction, args: ToolArgs) {
        self.builder.call(Some(id), tool, action, args);
        self.compacted = false;
    }

    fn function_call(&mut self, payload: &Value) {
        let (Some(name), Some(id)) = (
            Line::field(payload, "name"),
            Line::field(payload, "call_id"),
        ) else {
            return;
        };
        if Rollout::BOOKKEEPING_TOOLS.contains(&name) {
            return;
        }
        let args = match payload.get("arguments") {
            Some(Value::String(text)) => serde_json::from_str(text).unwrap_or_default(),
            Some(value) => value.clone(),
            None => Value::Null,
        };
        let workdir = Line::field(&args, "workdir");
        match name {
            "exec_command" => {
                let command = Line::field(&args, "cmd").unwrap_or_default();
                match Self::heredoc_patch(command) {
                    Some(patch) => self.patch(id, patch),
                    None => self.shell(id, name, command, workdir),
                }
            }
            "shell_command" => {
                let command = Line::field(&args, "command").unwrap_or_default();
                self.shell(id, name, command, workdir);
            }
            "shell" | "container.exec" => {
                let command = Self::script(args.get("command"));
                match Self::heredoc_patch(&command) {
                    Some(patch) => self.patch(id, patch),
                    None => self.shell(id, "shell", &command, workdir),
                }
            }
            "write_stdin" => {
                let process = match args.get("session_id") {
                    Some(Value::Number(number)) => number.to_string(),
                    Some(Value::String(text)) => text.clone(),
                    _ => return,
                };
                self.polls.insert(id.to_owned(), process);
            }
            "apply_patch" => {
                let patch = Line::field(&args, "input")
                    .or_else(|| Line::field(&args, "patch"))
                    .unwrap_or_default();
                self.patch(id, patch);
            }
            "view_image" => {
                let args = ToolArgs {
                    path: Line::field(&args, "path").map(|path| self.builder.relative(path)),
                    ..ToolArgs::default()
                };
                self.call(id, name, ToolAction::Read, args);
            }
            "spawn_agent" => self.call(id, name, ToolAction::Delegate, ToolArgs::default()),
            _ => self.call(id, name, ToolAction::Other, ToolArgs::default()),
        }
    }

    fn custom_tool_call(&mut self, payload: &Value) {
        let (Some(name), Some(id)) = (
            Line::field(payload, "name"),
            Line::field(payload, "call_id"),
        ) else {
            return;
        };
        if name == "apply_patch" {
            self.patch(id, Line::field(payload, "input").unwrap_or_default());
        } else if !Rollout::BOOKKEEPING_TOOLS.contains(&name) {
            self.call(id, name, ToolAction::Other, ToolArgs::default());
        }
    }

    fn local_shell_call(&mut self, payload: &Value) {
        let Some(id) = Line::field(payload, "call_id").or_else(|| Line::field(payload, "id"))
        else {
            return;
        };
        let action = payload.get("action").unwrap_or(&Value::Null);
        let command = Self::script(action.get("command"));
        let workdir = Line::field(action, "working_directory");
        self.shell(id, "local_shell", &command, workdir);
    }

    fn web_search(&mut self, payload: &Value) {
        let action = payload.get("action").unwrap_or(&Value::Null);
        let query = Line::field(action, "query")
            .or_else(|| Line::field(action, "url"))
            .map(|query| self.builder.redact(query));
        self.searches += 1;
        let id = format!("web_search_call {}", self.searches);
        self.call(
            &id,
            "web_search",
            ToolAction::Fetch,
            ToolArgs {
                query,
                ..ToolArgs::default()
            },
        );
        let finish = match Line::field(payload, "status") {
            None | Some("completed") => Finish::Succeeded,
            Some(_) => Finish::Interrupted,
        };
        self.builder.resolve(&id, finish, "", None, Vec::new());
    }

    fn shell(&mut self, id: &str, tool: &str, command: &str, workdir: Option<&str>) {
        let workdir = workdir.filter(|dir| !dir.is_empty()).map(|dir| {
            if TraceBuilder::is_absolute(dir) || self.cwd.is_empty() {
                dir.to_owned()
            } else {
                format!("{}/{dir}", self.cwd)
            }
        });
        if let Some(dir) = &workdir {
            self.builder.directory(dir);
        }
        let (args, action) = self.builder.shell(command);
        if workdir.is_some() && !self.cwd.is_empty() {
            self.builder.directory(&self.cwd);
        }
        self.call(id, tool, action, args);
    }

    fn script(argv: Option<&Value>) -> String {
        let argv: Vec<&str> = argv
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        match argv.as_slice() {
            [shell, flag, script]
                if ["bash", "sh", "zsh", "/bin/bash", "/bin/sh", "/bin/zsh"].contains(shell)
                    && ["-c", "-lc"].contains(flag) =>
            {
                (*script).to_owned()
            }
            _ => argv.join(" "),
        }
    }

    fn heredoc_patch(command: &str) -> Option<&str> {
        let command = command.trim_start();
        if !command.starts_with("apply_patch") && !command.starts_with("applypatch") {
            return None;
        }
        let start = command.find("*** Begin Patch")?;
        let end = command[start..]
            .find("*** End Patch")
            .map_or(command.len(), |end| start + end + "*** End Patch".len());
        Some(&command[start..end])
    }

    fn patch(&mut self, id: &str, patch: &str) {
        let files = PatchFile::parse(patch);
        let path = files.first().map(|file| self.builder.relative(&file.path));
        let changes = files
            .iter()
            .map(|file| {
                self.builder
                    .change(&file.path, file.created, &PatchFile::diff(&file.body), None)
            })
            .collect();
        self.patches.insert(id.to_owned(), changes);
        self.call(
            id,
            "apply_patch",
            ToolAction::Edit,
            ToolArgs {
                path,
                ..ToolArgs::default()
            },
        );
    }

    fn patch_end(&mut self, payload: &Value) {
        let Some(id) = Line::field(payload, "call_id") else {
            return;
        };
        let mut files: Vec<(&String, &Value)> = payload
            .get("changes")
            .and_then(Value::as_object)
            .into_iter()
            .flatten()
            .collect();
        files.sort_by(|a, b| a.0.cmp(b.0));
        let changes = files
            .into_iter()
            .filter_map(|(path, change)| {
                let (created, diff) = match Line::field(change, "type")? {
                    "add" => (
                        true,
                        Diff::created(Line::field(change, "content").unwrap_or_default()),
                    ),
                    "delete" => {
                        let removed =
                            Diff::created(Line::field(change, "content").unwrap_or_default());
                        (
                            false,
                            Diff {
                                removed: removed.added,
                                changed: removed.changed,
                                ..Diff::default()
                            },
                        )
                    }
                    "update" => (
                        false,
                        PatchFile::diff(Line::field(change, "unified_diff").unwrap_or_default()),
                    ),
                    _ => return None,
                };
                let path = Line::field(change, "move_path").unwrap_or(path);
                Some(self.builder.change(path, created, &diff, None))
            })
            .collect();
        let status = Line::field(payload, "status");
        self.patched.insert(
            id.to_owned(),
            PatchEnd {
                changes,
                success: payload.get("success").and_then(Value::as_bool) != Some(false)
                    && status != Some("failed"),
                declined: status == Some("declined"),
            },
        );
    }

    fn output(&mut self, id: &str, text: &str) {
        if let Some(process) = self.polls.remove(id) {
            self.poll(&process, text);
            return;
        }
        let Some(tool) = self.builder.pending_call(id).map(|call| call.tool.clone()) else {
            return;
        };
        let exec = Exec::parse(text);
        if let Some(exec) = exec
            && let Some(process) = exec.running
        {
            self.running.insert(
                process.to_owned(),
                Running {
                    call: id.to_owned(),
                    output: exec.output.to_owned(),
                    wall_ms: exec.wall_ms.unwrap_or_default(),
                },
            );
            return;
        }
        let (finish, output, wall_ms) = if Exec::interrupted(text) {
            (Finish::Interrupted, text.to_owned(), None)
        } else if let Some(exec) = exec
            && let Some(code) = exec.exit_code
        {
            (Self::exit(code), exec.output.to_owned(), exec.wall_ms)
        } else if let Some((code, output, wall_ms)) = Exec::legacy(text) {
            (Self::exit(code), output, wall_ms)
        } else if Rollout::SHELL_TOOLS.contains(&tool.as_str()) || tool == "apply_patch" {
            (Finish::Failed { exit_code: None }, text.to_owned(), None)
        } else {
            (Finish::Succeeded, text.to_owned(), None)
        };
        self.finish(id, finish, &output, wall_ms);
    }

    fn poll(&mut self, process: &str, text: &str) {
        let Some(exec) = Exec::parse(text) else {
            return;
        };
        let Some(running) = self.running.get_mut(process) else {
            return;
        };
        running.output.push_str(exec.output);
        running.wall_ms += exec.wall_ms.unwrap_or_default();
        let finish = if Exec::interrupted(text) {
            Finish::Interrupted
        } else if let Some(code) = exec.exit_code {
            Self::exit(code)
        } else {
            return;
        };
        if let Some(running) = self.running.remove(process) {
            self.finish(
                &running.call,
                finish,
                &running.output,
                Some(running.wall_ms),
            );
        }
    }

    fn finish(&mut self, id: &str, finish: Finish, output: &str, wall_ms: Option<u64>) {
        let fallback = self.patches.remove(id).unwrap_or_default();
        let (finish, changes) = match self.patched.remove(id) {
            Some(end) if end.declined => (Finish::Interrupted, Vec::new()),
            Some(end) if !end.success && finish == Finish::Succeeded => {
                (Finish::Failed { exit_code: None }, Vec::new())
            }
            Some(end) => (finish, end.changes),
            None => (finish, fallback),
        };
        self.builder.resolve(id, finish, output, wall_ms, changes);
    }

    fn exit(code: i32) -> Finish {
        if code == 0 {
            Finish::Succeeded
        } else {
            Finish::Failed {
                exit_code: Some(code),
            }
        }
    }

    fn text(value: Option<&Value>) -> String {
        match value {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(items)) => items
                .iter()
                .filter_map(|item| Line::field(item, "text"))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    fn trace(self) -> Result<Trace> {
        self.builder.finish("no Codex session found in rollout")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PatchFile {
    pub(crate) path: String,
    pub(crate) created: bool,
    pub(crate) body: String,
}

impl PatchFile {
    pub(crate) fn parse(patch: &str) -> Vec<Self> {
        let mut files: Vec<Self> = Vec::new();
        for line in patch.lines() {
            let header = |prefix: &str| line.strip_prefix(prefix).map(str::trim);
            if let Some(path) = header("*** Add File: ") {
                files.push(Self::new(path, true));
            } else if let Some(path) =
                header("*** Update File: ").or_else(|| header("*** Delete File: "))
            {
                files.push(Self::new(path, false));
            } else if let Some(path) = header("*** Move to: ") {
                if let Some(file) = files.last_mut() {
                    path.clone_into(&mut file.path);
                }
            } else if line.starts_with("*** ") {
            } else if let Some(file) = files.last_mut() {
                file.body.push_str(line);
                file.body.push('\n');
            }
        }
        files
    }

    pub(crate) fn diff(text: &str) -> Diff {
        let mut diff = Diff::unified(text);
        for header in text.lines().filter_map(|line| line.strip_prefix("@@")) {
            let context = match header.find("@@") {
                Some(end) => &header[end + 2..],
                None => header,
            };
            if !context.trim().is_empty() {
                diff.changed.push(context.trim().to_owned());
            }
        }
        diff
    }

    fn new(path: &str, created: bool) -> Self {
        Self {
            path: path.to_owned(),
            created,
            body: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use trodden_core::trace::{EventKind, ToolCall, ToolOutcome};

    use super::*;

    const CWD: &str = "/home/dev/shop";
    const SESSION: &str = "019f0a10-5c2e-7d41-9b8a-3e6f1c2d4a5b";

    #[derive(Debug, Default)]
    struct Session {
        lines: Vec<Value>,
        second: u32,
    }

    impl Session {
        fn new() -> Self {
            let mut session = Self::default();
            session.record(
                "session_meta",
                json!({"id": SESSION, "timestamp": "2026-06-07T12:18:19.788Z", "cwd": CWD,
                       "originator": "codex-tui", "cli_version": "0.137.0", "source": "cli",
                       "thread_source": "user", "model_provider": "openai",
                       "base_instructions": {"text": "You are Codex, a coding agent based on GPT-5."},
                       "git": {"commit_hash": "22f55037", "branch": "main"}}),
            );
            session
        }

        fn record(&mut self, kind: &str, payload: Value) -> &mut Self {
            self.second += 1;
            self.lines.push(json!({
                "timestamp": format!("2026-06-07T12:21:{:02}.584Z", self.second % 60),
                "type": kind,
                "payload": payload,
            }));
            self
        }

        fn turn(&mut self, turn: &str, prompt: &str) -> &mut Self {
            self.record("event_msg", json!({"type": "task_started", "turn_id": turn, "model_context_window": 258_400}))
                .record("response_item", json!({"type": "message", "role": "developer",
                    "content": [{"type": "input_text", "text": "<permissions instructions>\nFilesystem sandboxing defines which files can be read"}]}))
                .record("response_item", json!({"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": format!("<environment_context>\n  <cwd>{CWD}</cwd>\n  <shell>zsh</shell>\n</environment_context>")}]}))
                .record("turn_context", json!({"turn_id": turn, "cwd": CWD, "current_date": "2026-06-07",
                    "approval_policy": "on-request", "model": "gpt-5.5", "effort": "medium"}))
                .record("response_item", json!({"type": "message", "role": "user",
                    "content": [{"type": "input_text", "text": prompt}]}))
                .record("event_msg", json!({"type": "user_message", "message": prompt, "images": [], "local_images": [], "text_elements": []}))
        }

        fn exec(&mut self, call: &str, cmd: &str) -> &mut Self {
            let arguments = json!({"cmd": cmd, "workdir": CWD, "yield_time_ms": 10_000, "max_output_tokens": 20_000});
            self.record(
                "response_item",
                json!({"type": "function_call", "name": "exec_command",
                "arguments": arguments.to_string(), "call_id": call}),
            )
        }

        fn poll(&mut self, call: &str, process: u32) -> &mut Self {
            let arguments = json!({"session_id": process, "chars": "", "yield_time_ms": 1000, "max_output_tokens": 20_000});
            self.record(
                "response_item",
                json!({"type": "function_call", "name": "write_stdin",
                "arguments": arguments.to_string(), "call_id": call}),
            )
        }

        fn exited(&mut self, call: &str, code: i32, output: &str) -> &mut Self {
            self.output(call, &format!("Chunk ID: ba3f50\nWall time: 0.2500 seconds\nProcess exited with code {code}\nOriginal token count: 54\nOutput:\n{output}"))
        }

        fn running(&mut self, call: &str, process: u32) -> &mut Self {
            self.output(call, &format!("Chunk ID: 5f91e7\nWall time: 1.0017 seconds\nProcess running with session ID {process}\nOriginal token count: 14\nOutput:\n    Blocking waiting for file lock on build directory\n"))
        }

        fn output(&mut self, call: &str, output: &str) -> &mut Self {
            self.record(
                "response_item",
                json!({"type": "function_call_output", "call_id": call, "output": output}),
            )
        }

        fn patch(&mut self, call: &str, input: &str, changes: Value, exit: i32) -> &mut Self {
            self.record("response_item", json!({"type": "custom_tool_call", "status": "completed",
                    "call_id": call, "name": "apply_patch", "input": input}))
                .record("event_msg", json!({"type": "patch_apply_end", "call_id": call, "turn_id": "t",
                    "stdout": "Success. Updated the following files:\n", "stderr": "", "success": exit == 0,
                    "changes": changes, "status": if exit == 0 { "completed" } else { "failed" }}))
                .record("response_item", json!({"type": "custom_tool_call_output", "call_id": call,
                    "output": format!("Exit code: {exit}\nWall time: 0.5 seconds\nOutput:\nSuccess. Updated the following files:\n")}))
        }

        fn complete(&mut self, turn: &str) -> &mut Self {
            self.record(
                "event_msg",
                json!({"type": "task_complete", "turn_id": turn, "last_agent_message": "Done."}),
            )
        }

        fn text(&self) -> String {
            self.lines.iter().map(|line| format!("{line}\n")).collect()
        }

        fn trace(&self) -> Trace {
            Rollout::parse(&self.text(), &Redactor::with_home("/home/dev")).expect("rollout parses")
        }

        fn prompts(&self) -> Vec<String> {
            self.trace()
                .events
                .into_iter()
                .filter_map(|event| match event.kind {
                    EventKind::Prompt { summary } => Some(summary),
                    _ => None,
                })
                .collect()
        }

        fn calls(&self) -> Vec<ToolCall> {
            self.trace()
                .events
                .into_iter()
                .filter_map(|event| match event.kind {
                    EventKind::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect()
        }
    }

    #[test]
    fn prompts_come_from_user_messages_only() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Page 2 repeats the last product. Fix it.\nThe test is in test/")
            .record("response_item", json!({"type": "message", "role": "developer",
                "content": [{"type": "input_text", "text": "<trodden-memory>run npm test</trodden-memory>"}]}))
            .record("response_item", json!({"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "<turn_aborted>\nThe user interrupted the previous turn on purpose."}]}))
            .record("response_item", json!({"type": "message", "role": "user",
                "content": [{"type": "input_text", "text": "<hook_prompt hook_run_id=\"r1\">Trodden: the procedure recalled for this task</hook_prompt>"}]}));

        let trace = session.trace();

        assert_eq!(trace.session.as_str(), SESSION);
        assert_eq!(trace.harness.as_str(), "codex");
        assert_eq!(trace.cwd, "~/shop");
        assert_eq!(trace.model.as_deref(), Some("gpt-5.5"));
        assert_eq!(
            session.prompts(),
            ["Page 2 repeats the last product. Fix it."]
        );
    }

    #[test]
    fn old_rollouts_without_user_message_events_fall_back_to_user_messages() {
        let mut session = Session::new();
        session.turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug");
        session
            .lines
            .retain(|line| line["payload"]["type"] != "user_message");

        assert_eq!(session.prompts(), ["Fix the paging bug"]);
    }

    #[test]
    fn shell_commands_take_their_exit_code_from_the_output_header() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .exec("call_ok", "npm test")
            .exec("call_fmt", "cargo fmt --all -- --check")
            .exec("call_read", "cat src/paginate.js")
            .exited("call_ok", 0, "1 passing\n")
            .exited(
                "call_fmt",
                1,
                "Diff in /home/dev/shop/src/lib.rs at line 3:\n",
            )
            .exited("call_read", 0, "export function paginate(items) {}\n");

        let calls = session.calls();

        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].tool, "exec_command");
        assert_eq!(calls[0].action, ToolAction::Run);
        assert_eq!(calls[0].args.command.as_deref(), Some("npm test"));
        assert_eq!(calls[0].outcome, ToolOutcome::Succeeded);
        assert_eq!(calls[0].duration_ms, Some(250));
        assert_eq!(calls[1].outcome, ToolOutcome::Failed { exit_code: Some(1) });
        assert_eq!(calls[2].action, ToolAction::Read);
    }

    #[test]
    fn long_running_commands_finish_through_write_stdin() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Make clippy pass")
            .exec("call_test", "cargo test --workspace")
            .exec("call_clippy", "cargo clippy --workspace --all-targets -- -D warnings")
            .running("call_test", 90_670)
            .running("call_clippy", 70_738)
            .poll("call_poll_1", 70_738)
            .output("call_poll_1", "Chunk ID: 1c0223\nWall time: 1.0027 seconds\nProcess running with session ID 70738\nOriginal token count: 4\nOutput:\n    Checking shop v0.1.0\n")
            .poll("call_poll_2", 90_670)
            .poll("call_poll_3", 70_738)
            .exited("call_poll_2", 0, "test result: ok. 42 passed; 0 failed\n")
            .exited("call_poll_3", 101, "error: this `if` statement can be collapsed\n  --> src/router.rs:31:13\n");

        let calls = session.calls();

        assert_eq!(calls.len(), 2, "polls are not calls of their own");
        assert_eq!(calls[0].outcome, ToolOutcome::Succeeded);
        assert_eq!(calls[0].duration_ms, Some(1252));
        assert_eq!(
            calls[1].outcome,
            ToolOutcome::Failed {
                exit_code: Some(101)
            }
        );
        assert_eq!(calls[1].duration_ms, Some(2255));
        assert!(
            calls[1]
                .error
                .as_deref()
                .is_some_and(|error| error.contains("statement can be collapsed")),
            "{:?}",
            calls[1].error
        );
    }

    #[test]
    fn commands_still_running_when_the_turn_is_interrupted_stay_interrupted() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Run the slow tests")
            .exec("call_slow", "npm run test:slow")
            .running("call_slow", 23_031)
            .record("event_msg", json!({"type": "turn_aborted", "turn_id": "019f0a11-0000-7000-8000-000000000001", "reason": "interrupted"}))
            .exec("call_rejected", "rm -rf build")
            .output("call_rejected", "exec command rejected by user");

        let calls = session.calls();

        assert_eq!(calls[0].outcome, ToolOutcome::Interrupted);
        assert_eq!(calls[1].outcome, ToolOutcome::Interrupted);
    }

    #[test]
    fn patches_record_their_changes_with_line_counts() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .patch(
                "call_patch",
                "*** Begin Patch\n*** Update File: src/paginate.js\n@@ export function paginate\n-  const end = start + perPage + 1;\n+  const end = start + perPage;\n*** Add File: test/paging.test.js\n+import { paginate } from '../src/paginate.js';\n+test('page 2', () => {});\n*** End Patch",
                json!({
                    "/home/dev/shop/test/paging.test.js": {"type": "add", "content": "import { paginate } from '../src/paginate.js';\ntest('page 2', () => {});\n"},
                    "/home/dev/shop/src/paginate.js": {"type": "update", "unified_diff": "@@ -3,3 +3,3 @@ export function paginate(items, page, perPage) {\n   const start = (page - 1) * perPage;\n-  const end = start + perPage + 1;\n+  const end = start + perPage;\n   return items.slice(start, end);\n", "move_path": null},
                    "/home/dev/shop/src/old.js": {"type": "delete", "content": "a\nb\n"}
                }),
                0,
            );

        let calls = session.calls();

        assert_eq!(calls.len(), 1);
        let call = &calls[0];
        assert_eq!(call.tool, "apply_patch");
        assert_eq!(call.action, ToolAction::Edit);
        assert_eq!(call.args.path.as_deref(), Some("src/paginate.js"));
        assert_eq!(call.outcome, ToolOutcome::Succeeded);
        let changes: Vec<(&str, bool, u32, u32)> = call
            .changes
            .iter()
            .map(|change| {
                (
                    change.path.as_str(),
                    change.created,
                    change.lines_added,
                    change.lines_removed,
                )
            })
            .collect();
        assert_eq!(
            changes,
            [
                ("src/old.js", false, 0, 2),
                ("src/paginate.js", false, 1, 1),
                ("test/paging.test.js", true, 2, 0)
            ]
        );
        assert_eq!(call.changes[1].symbols, ["paginate"]);
    }

    #[test]
    fn failed_patches_change_nothing() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .patch("call_patch", "*** Begin Patch\n*** Update File: src/paginate.js\n@@\n-x\n+y\n*** End Patch", json!({}), 1)
            .record("response_item", json!({"type": "custom_tool_call", "status": "completed", "call_id": "call_bad",
                "name": "apply_patch", "input": "*** Begin Patch\n*** Update File: src/missing.js\n@@\n-x\n+y\n*** End Patch"}))
            .record("response_item", json!({"type": "custom_tool_call_output", "call_id": "call_bad",
                "output": "apply_patch verification failed: Failed to find expected lines in src/missing.js"}));

        let calls = session.calls();

        assert_eq!(calls[0].outcome, ToolOutcome::Failed { exit_code: Some(1) });
        assert!(calls[0].changes.is_empty());
        assert_eq!(calls[1].outcome, ToolOutcome::Failed { exit_code: None });
        assert!(calls[1].changes.is_empty());
    }

    #[test]
    fn patches_without_an_apply_event_fall_back_to_the_patch_body() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .exec("call_heredoc", "apply_patch <<'EOF'\n*** Begin Patch\n*** Update File: src/paginate.js\n*** Move to: src/paging.js\n@@\n-  const end = start + perPage + 1;\n+  const end = start + perPage;\n*** End Patch\nEOF")
            .exited("call_heredoc", 0, "Success. Updated the following files:\nM src/paging.js\n");

        let calls = session.calls();

        assert_eq!(calls[0].tool, "apply_patch");
        assert_eq!(calls[0].action, ToolAction::Edit);
        assert_eq!(calls[0].changes[0].path, "src/paging.js");
        assert_eq!(
            (
                calls[0].changes[0].lines_added,
                calls[0].changes[0].lines_removed
            ),
            (1, 1)
        );
    }

    #[test]
    fn bookkeeping_is_ignored_and_other_tools_are_kept() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .record("response_item", json!({"type": "function_call", "name": "update_plan",
                "arguments": "{\"plan\":[{\"step\":\"Fix\",\"status\":\"in_progress\"}]}", "call_id": "call_plan"}))
            .output("call_plan", "Plan updated")
            .record("response_item", json!({"type": "function_call", "name": "view_image",
                "arguments": "{\"path\":\"/home/dev/shop/docs/page.png\"}", "call_id": "call_image"}))
            .record("response_item", json!({"type": "function_call_output", "call_id": "call_image",
                "output": [{"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}]}))
            .record("response_item", json!({"type": "web_search_call", "status": "completed",
                "action": {"type": "search", "query": "paginate off by one"}}))
            .record("response_item", json!({"type": "function_call", "name": "mcp__docs__lookup",
                "arguments": "{}", "call_id": "call_mcp"}))
            .output("call_mcp", "{\"ok\":true}");

        let calls = session.calls();
        let summary: Vec<(&str, ToolAction, &ToolOutcome)> = calls
            .iter()
            .map(|call| (call.tool.as_str(), call.action, &call.outcome))
            .collect();

        assert_eq!(
            summary,
            [
                ("view_image", ToolAction::Read, &ToolOutcome::Succeeded),
                ("web_search", ToolAction::Fetch, &ToolOutcome::Succeeded),
                (
                    "mcp__docs__lookup",
                    ToolAction::Other,
                    &ToolOutcome::Succeeded
                ),
            ]
        );
        assert_eq!(calls[0].args.path.as_deref(), Some("docs/page.png"));
        assert_eq!(calls[1].args.query.as_deref(), Some("paginate off by one"));
    }

    #[test]
    fn a_new_turn_directory_becomes_a_cd_prefix() {
        let mut session = Session::new();
        session.turn("019f0a11-0000-7000-8000-000000000001", "Fix the web tests");
        session
            .record("response_item", json!({"type": "function_call", "name": "exec_command",
                "arguments": json!({"cmd": "npm test", "workdir": "web"}).to_string(), "call_id": "call_web"}))
            .exited("call_web", 0, "ok\n")
            .exec("call_root", "npm run lint")
            .exited("call_root", 0, "ok\n");

        let calls = session.calls();

        assert_eq!(calls[0].args.command.as_deref(), Some("cd web && npm test"));
        assert_eq!(calls[1].args.command.as_deref(), Some("npm run lint"));
    }

    #[test]
    fn rolled_back_turns_are_dropped() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .exec("call_kept", "npm test")
            .exited("call_kept", 0, "ok\n")
            .complete("019f0a11-0000-7000-8000-000000000001")
            .turn("019f0a12-0000-7000-8000-000000000002", "Implemnt the plan")
            .exec("call_dropped", "npm run build")
            .record("event_msg", json!({"type": "turn_aborted", "turn_id": "019f0a12-0000-7000-8000-000000000002", "reason": "interrupted"}))
            .record("event_msg", json!({"type": "thread_rolled_back", "num_turns": 1}))
            .turn("019f0a13-0000-7000-8000-000000000003", "Implement the plan")
            .record("event_msg", json!({"type": "thread_rolled_back", "num_turns": 9}))
            .turn("019f0a14-0000-7000-8000-000000000004", "Implement the plan in PLAN.md");

        assert_eq!(session.prompts(), ["Implement the plan in PLAN.md"]);
        assert!(session.calls().is_empty());
    }

    #[test]
    fn forked_rollouts_skip_the_copied_parent_history() {
        let mut session = Session::default();
        session
            .record(
                "session_meta",
                json!({"id": "019f0b00-0000-7000-8000-000000000000", "cwd": CWD,
                "source": "cli", "forked_from_id": SESSION}),
            )
            .record(
                "session_meta",
                json!({"id": SESSION, "cwd": CWD, "source": "cli"}),
            )
            .turn("019f0a11-0000-7000-8000-000000000001", "Parent prompt")
            .exec("call_parent", "npm test")
            .exited("call_parent", 0, "ok\n")
            .turn("019f0b01-0000-7000-8000-000000000000", "Fork prompt");

        let trace = session.trace();

        assert_eq!(
            trace.session.as_str(),
            "019f0b00-0000-7000-8000-000000000000"
        );
        assert_eq!(session.prompts(), ["Fork prompt"]);
        assert!(session.calls().is_empty());
    }

    #[test]
    fn compactions_are_counted_once() {
        let mut session = Session::new();
        session
            .turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug")
            .record(
                "compacted",
                json!({"message": "Summary of the work so far", "replacement_history": []}),
            )
            .record("event_msg", json!({"type": "context_compacted"}))
            .record(
                "event_msg",
                json!({"type": "task_started", "turn_id": "019f0a12-0000-7000-8000-000000000002"}),
            )
            .record("compacted", json!({"message": "Manual summary"}));

        let compactions: Vec<bool> = session
            .trace()
            .events
            .into_iter()
            .filter_map(|event| match event.kind {
                EventKind::Compaction { automatic } => Some(automatic),
                _ => None,
            })
            .collect();

        assert_eq!(compactions, [true, false]);
    }

    #[test]
    fn subagent_rollouts_are_left_to_their_parent() {
        let meta =
            json!({"timestamp": "2026-06-07T12:21:17.584Z", "type": "session_meta", "payload": {
            "id": "019f0c00-0000-7000-8000-000000000000", "cwd": CWD,
            "source": {"subagent": {"thread_spawn": {"parent_thread_id": SESSION, "depth": 1}}},
            "thread_source": "subagent"}})
            .to_string();
        let redactor = Redactor::with_home("/home/dev");

        assert!(Rollout::is_subagent(&meta));
        assert!(!Rollout::is_subagent(&Session::new().text()));
        assert!(Rollout::parse(&meta, &redactor).is_err());
    }

    #[test]
    fn malformed_lines_are_skipped() {
        let mut session = Session::new();
        session.turn("019f0a11-0000-7000-8000-000000000001", "Fix the paging bug");
        let text = format!(
            "{}not json\n[1,2]\n{{\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call\"}}}}\n{{\"type\":\"event_msg\",\"payload\":7}}\n{{\"timestamp\":\"x\",\"type\":\"response_item\",\"payload\":{{\"type\":\"function_call_output\",\"call_id\":\"nobody\",\"output\":\"Process exited with code 3\"}}}}\n{{\"type\":\"event_msg\",\"payl",
            session.text()
        );

        let trace =
            Rollout::parse(&text, &Redactor::with_home("/home/dev")).expect("rollout parses");

        assert_eq!(trace.events.len(), 1);
        assert!(Rollout::parse("garbage\n", &Redactor::with_home("/home/dev")).is_err());
    }

    #[test]
    fn working_directory_is_the_session_root() {
        assert_eq!(
            Rollout::working_directory(&Session::new().text()),
            Some(PathBuf::from(CWD))
        );
    }

    #[test]
    fn hook_outputs_reveal_failures() {
        let redactor = Redactor::with_home("/home/dev");

        assert!(Rollout::failed(
            "cargo test",
            "error[E0425]: cannot find value `total` in this scope\n --> src/lib.rs:3:5\n",
            &redactor
        ));
        assert!(Rollout::failed(
            "make",
            "Exit code: 2\nWall time: 1 seconds\nOutput:\nboom\n",
            &redactor
        ));
        assert!(!Rollout::failed(
            "cargo test",
            "test result: ok. 3 passed; 0 failed\n",
            &redactor
        ));
        assert!(!Rollout::failed(
            "cat src/errors.rs",
            "error: unknown flag\n",
            &redactor
        ));
        assert!(!Rollout::failed(
            "cargo build",
            "Process exited with code 0\nOutput:\nerror: looks bad but passed\n",
            &redactor
        ));
    }

    #[test]
    fn exec_headers_parse_and_plain_text_does_not() {
        let exec = Exec::parse("Chunk ID: ec963a\nWall time: 0.0000 seconds\nProcess exited with code 101\nOriginal token count: 268\nOutput:\n    Checking\n")
            .expect("header");

        assert_eq!(exec.exit_code, Some(101));
        assert_eq!(exec.wall_ms, Some(0));
        assert_eq!(exec.output, "    Checking\n");
        assert_eq!(Exec::parse("Plan updated"), None);
        assert_eq!(
            Exec::legacy(
                "{\"output\":\"boom\",\"metadata\":{\"exit_code\":2,\"duration_seconds\":1.5}}"
            ),
            Some((2, "boom".to_owned(), Some(1500)))
        );
    }
}
