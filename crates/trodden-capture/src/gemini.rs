use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::Value;
use trodden_core::{
    Trace,
    trace::{FileChange, ToolAction, ToolArgs},
};
use trodden_redact::Redactor;

use crate::{
    builder::{Finish, Prompts, TraceBuilder},
    diff::Diff,
};

pub const HARNESS: &str = "gemini";

#[derive(Debug)]
pub struct Transcript;

impl Transcript {
    const PROMPTS: Prompts = Prompts {
        skipped: &[
            "<hook_context>",
            "<session_context>",
            "<state_snapshot>",
            "/",
            "?",
        ],
        pasted: None,
    };

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "write_todos",
        "ask_user",
        "enter_plan_mode",
        "exit_plan_mode",
        "update_topic",
        "complete_task",
        "activate_skill",
        "save_memory",
        "get_internal_docs",
        "tracker_create_task",
        "tracker_update_task",
        "tracker_get_task",
        "tracker_list_tasks",
        "tracker_add_dependency",
        "tracker_visualize",
    ];

    pub fn session(text: &str) -> Option<String> {
        Replay::new(text).session
    }

    pub fn directories(text: &str) -> Vec<String> {
        Replay::new(text).directories
    }

    pub fn parse(text: &str, cwd: Option<&str>, redactor: &Redactor) -> Result<Trace> {
        let replay = Replay::new(text);
        if replay.subagent {
            bail!("this is a Gemini CLI subagent transcript, not a session");
        }
        let mut reader = Reader {
            builder: TraceBuilder::new(HARNESS, redactor),
            root: cwd.unwrap_or_default().to_owned(),
        };
        if let Some(session) = &replay.session {
            reader.builder.session(session);
        }
        if let Some(cwd) = cwd {
            reader.builder.directory(cwd);
        }
        if let Some(start) = &replay.start {
            reader.builder.time(start);
        }
        for entry in &replay.order {
            match entry {
                Entry::Compaction => reader.builder.compaction(true),
                Entry::Message(id) => {
                    if let Some(message) = replay.messages.get(id) {
                        reader.push(message);
                    }
                }
            }
        }
        reader
            .builder
            .finish("no Gemini CLI session found in transcript")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellResult {
    pub output: String,
    pub exit_code: Option<i32>,
    pub failed: bool,
    pub interrupted: bool,
}

impl ShellResult {
    const TRAILERS: &[&str] = &[
        "Exit Code: ",
        "Signal: ",
        "Background PIDs: ",
        "Process Group PGID: ",
    ];

    pub fn parse(text: &str) -> Self {
        let text = text
            .split_once("\n\n<hook_context>")
            .map_or(text, |(before, _)| before);
        let text = text.trim();
        let text = text
            .strip_prefix("<untrusted_context>")
            .map_or(text, |rest| {
                rest.rsplit_once("</untrusted_context>")
                    .map_or(rest, |(inside, _)| inside)
            })
            .trim();
        if text.starts_with("Command was cancelled")
            || text.starts_with("Command was automatically cancelled")
        {
            return Self {
                output: text.to_owned(),
                exit_code: None,
                failed: false,
                interrupted: true,
            };
        }
        let mut lines: Vec<&str> = text.lines().collect();
        let mut exit_code = None;
        let mut signalled = false;
        while let Some(last) = lines.last() {
            let Some(trailer) = Self::TRAILERS.iter().find(|t| last.starts_with(**t)) else {
                break;
            };
            let value = last[trailer.len()..].trim();
            match *trailer {
                "Exit Code: " => exit_code = value.parse::<i32>().ok(),
                "Signal: " => signalled = !value.is_empty(),
                _ => {}
            }
            lines.pop();
        }
        let body = lines.join("\n");
        let output = body.strip_prefix("Output: ").unwrap_or(&body);
        let output = if output == "(empty)" { "" } else { output };
        Self {
            output: output.to_owned(),
            failed: exit_code.is_some_and(|code| code != 0) || signalled,
            exit_code,
            interrupted: false,
        }
    }
}

#[derive(Debug)]
enum Entry {
    Message(String),
    Compaction,
}

// Checkpoints only add messages, never replace them: compression and masking would erase
// learnable steps.
#[derive(Debug, Default)]
struct Replay {
    session: Option<String>,
    start: Option<String>,
    directories: Vec<String>,
    subagent: bool,
    order: Vec<Entry>,
    messages: HashMap<String, Value>,
}

impl Replay {
    fn new(text: &str) -> Self {
        let mut replay = Self::default();
        let trimmed = text.trim_start();
        if trimmed.starts_with('{')
            && let Ok(Value::Object(legacy)) = serde_json::from_str::<Value>(trimmed)
            && legacy.get("messages").is_some_and(Value::is_array)
        {
            replay.record(Value::Object(legacy));
            return replay;
        }
        for line in text.lines() {
            if let Ok(record @ Value::Object(_)) = serde_json::from_str::<Value>(line) {
                replay.record(record);
            }
        }
        replay
    }

    fn record(&mut self, record: Value) {
        if let Some(target) = record.get("$rewindTo").and_then(Value::as_str) {
            self.rewind(target);
        } else if let Some(set) = record.get("$set") {
            self.metadata(set);
            if let Some(messages) = set.get("messages").and_then(Value::as_array) {
                self.checkpoint(messages);
            }
        } else if let Some(id) = record.get("id").and_then(Value::as_str) {
            let id = id.to_owned();
            if self.messages.insert(id.clone(), record).is_none() {
                self.order.push(Entry::Message(id));
            }
        } else if record.get("sessionId").is_some() {
            self.metadata(&record);
            for message in record
                .get("messages")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(id) = message.get("id").and_then(Value::as_str)
                    && self
                        .messages
                        .insert(id.to_owned(), message.clone())
                        .is_none()
                {
                    self.order.push(Entry::Message(id.to_owned()));
                }
            }
        }
    }

    fn metadata(&mut self, fields: &Value) {
        if self.session.is_none()
            && let Some(session) = fields.get("sessionId").and_then(Value::as_str)
        {
            self.session = Some(session.to_owned());
        }
        if self.start.is_none()
            && let Some(start) = fields.get("startTime").and_then(Value::as_str)
        {
            self.start = Some(start.to_owned());
        }
        if let Some(directories) = fields.get("directories").and_then(Value::as_array) {
            self.directories = directories
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect();
        }
        if let Some(kind) = fields.get("kind").and_then(Value::as_str) {
            self.subagent = kind == "subagent";
        }
    }

    fn rewind(&mut self, target: &str) {
        let at = self
            .order
            .iter()
            .position(|entry| matches!(entry, Entry::Message(id) if id == target))
            .unwrap_or(0);
        for entry in self.order.drain(at..) {
            if let Entry::Message(id) = entry {
                self.messages.remove(&id);
            }
        }
    }

    fn checkpoint(&mut self, messages: &[Value]) {
        let added: Vec<&Value> = messages
            .iter()
            .filter(|message| {
                message
                    .get("id")
                    .and_then(Value::as_str)
                    .is_some_and(|id| !self.messages.contains_key(id))
            })
            .collect();
        let compressed = added.iter().any(|message| {
            message.get("type").and_then(Value::as_str) == Some("user")
                && Parts::texts(message.get("content"))
                    .iter()
                    .any(|text| text.contains("<state_snapshot>"))
        });
        if compressed {
            self.order.push(Entry::Compaction);
        }
        for message in added {
            if let Some(id) = message.get("id").and_then(Value::as_str) {
                self.messages.insert(id.to_owned(), message.clone());
                self.order.push(Entry::Message(id.to_owned()));
            }
        }
    }
}

#[derive(Debug)]
struct Parts;

impl Parts {
    fn list(content: Option<&Value>) -> Vec<&Value> {
        match content {
            Some(Value::Array(parts)) => parts.iter().collect(),
            Some(part @ (Value::Object(_) | Value::String(_))) => vec![part],
            _ => Vec::new(),
        }
    }

    fn texts(content: Option<&Value>) -> Vec<&str> {
        Self::list(content)
            .into_iter()
            .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
            .filter_map(|part| match part {
                Value::String(text) => Some(text.as_str()),
                _ => part.get("text").and_then(Value::as_str),
            })
            .collect()
    }

    fn has_function_response(content: Option<&Value>) -> bool {
        Self::list(content)
            .iter()
            .any(|part| part.get("functionResponse").is_some())
    }

    fn response(result: Option<&Value>) -> (String, bool) {
        let mut text = Vec::new();
        let mut error = false;
        for part in Self::list(result) {
            let Some(response) = part.get("functionResponse").and_then(|f| f.get("response"))
            else {
                if let Some(plain) = part.get("text").and_then(Value::as_str) {
                    text.push(plain.to_owned());
                }
                continue;
            };
            if let Some(output) = response.get("output") {
                text.push(Self::string(output));
            } else if let Some(failure) = response.get("error") {
                error = true;
                text.push(Self::string(failure));
            }
        }
        (text.join("\n"), error)
    }

    fn string(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            Value::Object(_) => value
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| value.to_string(), str::to_owned),
            other => other.to_string(),
        }
    }
}

struct Reader<'a> {
    builder: TraceBuilder<'a>,
    root: String,
}

impl Reader<'_> {
    fn push(&mut self, message: &Value) {
        if let Some(at) = message.get("timestamp").and_then(Value::as_str) {
            self.builder.time(at);
        }
        match message.get("type").and_then(Value::as_str) {
            Some("user") => self.push_user(message),
            Some("gemini") => self.push_gemini(message),
            _ => {}
        }
    }

    fn push_user(&mut self, message: &Value) {
        let content = message.get("content");
        if Parts::has_function_response(content) {
            return;
        }
        let shown = message
            .get("displayContent")
            .filter(|shown| !shown.is_null());
        let texts = Parts::texts(shown.or(content));
        let typed: Vec<&str> = texts
            .into_iter()
            .filter(|text| !text.trim_start().starts_with("<hook_context>"))
            .collect();
        if let Some(first) = typed.first() {
            self.builder.prompt(first, &Transcript::PROMPTS);
        }
    }

    fn push_gemini(&mut self, message: &Value) {
        if let Some(model) = message.get("model").and_then(Value::as_str) {
            self.builder.model(model);
        }
        let calls = message
            .get("toolCalls")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or_default();
        for (index, call) in calls.iter().enumerate() {
            self.push_call(call, index);
        }
    }

    fn push_call(&mut self, call: &Value, index: usize) {
        let Some(name) = call.get("name").and_then(Value::as_str) else {
            return;
        };
        if Transcript::BOOKKEEPING_TOOLS.contains(&name) {
            return;
        }
        let args = call.get("args").cloned().unwrap_or_default();
        let Some((action, tool_args)) = self.tool_call(name, &args) else {
            return;
        };
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .map_or_else(|| format!("{name}-{index}"), str::to_owned);
        if self.builder.is_pending(&id) {
            return;
        }
        if let Some(at) = call.get("timestamp").and_then(Value::as_str) {
            self.builder.time(at);
        }
        self.builder.call(Some(&id), name, action, tool_args);
        self.resolve(&id, name, action, &args, call);
    }

    fn tool_call(&mut self, name: &str, args: &Value) -> Option<(ToolAction, ToolArgs)> {
        let field = |key: &str| args.get(key).and_then(Value::as_str);
        let mut tool_args = ToolArgs::default();
        let action = match name {
            "run_shell_command" => {
                let command = field("command").unwrap_or_default();
                let (shell, action) = match field("dir_path").filter(|dir| !dir.is_empty()) {
                    Some(dir) => self.shell_in(dir, command),
                    None => self.builder.shell(command),
                };
                tool_args = shell;
                action
            }
            "replace" | "edit" | "write_file" => {
                tool_args.path = field("file_path").map(|p| self.builder.relative(p));
                ToolAction::Edit
            }
            "read_file" => {
                tool_args.path = field("file_path")
                    .or_else(|| field("absolute_path"))
                    .map(|p| self.builder.relative(p));
                ToolAction::Read
            }
            "list_directory" => {
                tool_args.path = field("dir_path")
                    .or_else(|| field("path"))
                    .map(|p| self.builder.relative(p));
                ToolAction::Read
            }
            "read_many_files" => ToolAction::Read,
            "grep_search" | "search_file_content" | "glob" => {
                tool_args.pattern =
                    field("pattern").map(|p| self.builder.search_pattern(p, name == "glob"));
                tool_args.path = field("dir_path")
                    .or_else(|| field("path"))
                    .map(|p| self.builder.relative(p));
                ToolAction::Search
            }
            "web_fetch" => {
                tool_args.url = field("url").map(|u| self.builder.redact(u));
                if tool_args.url.is_none() {
                    tool_args.query = field("prompt").map(|q| self.builder.redact(q));
                }
                ToolAction::Fetch
            }
            "google_web_search" => {
                tool_args.query = field("query").map(|q| self.builder.redact(q));
                ToolAction::Fetch
            }
            "invoke_agent" | "delegate_to_agent" => ToolAction::Delegate,
            _ => ToolAction::Other,
        };
        Some((action, tool_args))
    }

    fn shell_in(&mut self, dir: &str, command: &str) -> (ToolArgs, ToolAction) {
        if self.root.is_empty() {
            return self.builder.shell(command);
        }
        let here = if TraceBuilder::is_absolute(dir) {
            dir.to_owned()
        } else {
            format!("{}/{}", self.root.trim_end_matches(['/', '\\']), dir)
        };
        self.builder.directory(&here);
        let shell = self.builder.shell(command);
        let root = self.root.clone();
        self.builder.directory(&root);
        shell
    }

    fn resolve(&mut self, id: &str, name: &str, action: ToolAction, args: &Value, call: &Value) {
        let (text, errored) = Parts::response(call.get("result"));
        let status = call
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let (finish, output) = match status {
            "cancelled" => (Finish::Interrupted, text),
            "error" | "success" if errored || status == "error" => {
                if Self::denied(&text) {
                    (Finish::Interrupted, text)
                } else {
                    (Finish::Failed { exit_code: None }, text)
                }
            }
            "success" if name == "run_shell_command" => {
                let shell = ShellResult::parse(&text);
                let finish = if shell.interrupted {
                    Finish::Interrupted
                } else if shell.failed {
                    Finish::Failed {
                        exit_code: shell.exit_code,
                    }
                } else {
                    Finish::Succeeded
                };
                (finish, shell.output)
            }
            "success" => (Finish::Succeeded, text),
            _ => return,
        };
        let changes = if action == ToolAction::Edit && finish == Finish::Succeeded {
            self.file_change(name, args, call.get("resultDisplay"))
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        self.builder.resolve(id, finish, &output, None, changes);
    }

    fn denied(text: &str) -> bool {
        text.contains("denied by policy")
            || text.starts_with("User denied")
            || text.contains("did not allow")
    }

    fn file_change(&self, name: &str, args: &Value, display: Option<&Value>) -> Option<FileChange> {
        let field = |key: &str| args.get(key).and_then(Value::as_str);
        if let Some(display) = display.filter(|display| display.get("fileDiff").is_some()) {
            let path = display
                .get("filePath")
                .and_then(Value::as_str)
                .or_else(|| field("file_path"))?;
            let original = display.get("originalContent").and_then(Value::as_str);
            let created = display.get("isNewFile").and_then(Value::as_bool) == Some(true)
                || display.get("originalContent").is_some_and(Value::is_null);
            let diff = Diff::unified(
                display
                    .get("fileDiff")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            );
            return Some(self.builder.change(path, created, &diff, original));
        }
        let path = field("file_path")?;
        let diff = if name == "write_file" {
            Diff::created(field("content").unwrap_or_default())
        } else {
            Diff::replaced(
                field("old_string").unwrap_or_default(),
                field("new_string").unwrap_or_default(),
            )
        };
        Some(self.builder.change(path, false, &diff, None))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use trodden_core::trace::{EventKind, ToolCall, ToolOutcome};

    use super::*;

    const ROOT: &str = "/home/dev/shop";

    #[derive(Debug)]
    struct Session;

    impl Session {
        fn text(lines: &[Value]) -> String {
            lines.iter().map(|line| format!("{line}\n")).collect()
        }

        fn metadata() -> Value {
            json!({"sessionId": "8f0c2a3e-6d1b-4c55-9a7e-1f2b3c4d5e6f", "projectHash": "3b1f", "startTime": "2026-10-09T10:12:20.001Z", "lastUpdated": "2026-10-09T10:12:20.001Z", "kind": "main"})
        }

        fn user(id: &str, text: &str) -> Value {
            json!({"id": id, "timestamp": "2026-10-09T10:12:31.500Z", "type": "user", "content": [{"text": text}]})
        }

        fn gemini(id: &str, calls: Value) -> Value {
            json!({"id": id, "timestamp": "2026-10-09T10:12:35.010Z", "type": "gemini", "content": "", "model": "gemini-3-pro-preview", "toolCalls": calls})
        }

        fn call(id: &str, name: &str, args: Value, status: &str, response: Value) -> Value {
            json!({"id": id, "name": name, "args": args, "status": status, "timestamp": "2026-10-09T10:13:02.120Z",
                   "result": [{"functionResponse": {"id": id, "name": name, "response": response}}]})
        }

        fn shell(id: &str, command: &str, output: &str) -> Value {
            Self::call(
                id,
                "run_shell_command",
                json!({"command": command}),
                "success",
                json!({"output": format!("<untrusted_context>\n{output}\n</untrusted_context>")}),
            )
        }

        fn parse(lines: &[Value]) -> Trace {
            Transcript::parse(
                &Self::text(lines),
                Some(ROOT),
                &Redactor::with_home("/home/dev"),
            )
            .expect("valid transcript")
        }

        fn prompts(trace: &Trace) -> Vec<String> {
            trace
                .events
                .iter()
                .filter_map(|event| match &event.kind {
                    EventKind::Prompt { summary } => Some(summary.clone()),
                    _ => None,
                })
                .collect()
        }

        fn calls(trace: &Trace) -> Vec<ToolCall> {
            trace
                .events
                .iter()
                .filter_map(|event| match &event.kind {
                    EventKind::ToolCall(call) => Some(call.clone()),
                    _ => None,
                })
                .collect()
        }
    }

    #[test]
    fn real_prompts_are_kept_and_injected_text_is_not() {
        let trace = Session::parse(&[
            Session::metadata(),
            json!({"id": "env", "timestamp": "2026-10-09T10:12:21Z", "type": "user", "content": [{"text": "<session_context>\nThis is the Gemini CLI."}]}),
            json!({"id": "a1", "timestamp": "2026-10-09T10:12:31Z", "type": "user",
                   "content": [{"text": "fix the paging bug\n@src/paginate.js contents"}, {"text": "<hook_context>&lt;trodden-memory&gt;</hook_context>"}],
                   "displayContent": [{"text": "fix the paging bug"}]}),
            Session::user("a2", "/compress"),
            Session::user(
                "a3",
                "Trodden: the procedure recalled for this task is checked with `npm test`, which has not run since your last change. Run it once before finishing.",
            ),
            json!({"id": "a4", "timestamp": "2026-10-09T10:14:00Z", "type": "user", "content": [{"functionResponse": {"id": "x", "name": "read_file", "response": {"output": "text"}}}]}),
            json!({"id": "a5", "timestamp": "2026-10-09T10:15:00Z", "type": "user", "content": "now add a test for it"}),
        ]);

        assert_eq!(
            Session::prompts(&trace),
            ["fix the paging bug", "now add a test for it"]
        );
        assert_eq!(
            trace.session.as_str(),
            "8f0c2a3e-6d1b-4c55-9a7e-1f2b3c4d5e6f"
        );
        assert_eq!(trace.harness.as_str(), HARNESS);
        assert_eq!(trace.cwd, "~/shop");
    }

    #[test]
    fn shell_outcomes_come_from_the_exit_code_line() {
        let trace = Session::parse(&[
            Session::metadata(),
            Session::user("a1", "fix the clippy lint"),
            Session::gemini(
                "b1",
                json!([
                    Session::shell(
                        "c1",
                        "cargo clippy -- -D warnings",
                        "Output: error: this `if` has identical blocks\nerror: could not compile `shop`\nExit Code: 101\nProcess Group PGID: 51234"
                    ),
                    Session::shell(
                        "c2",
                        "cargo test",
                        "Output: test result: ok. 3 passed; 0 failed\nProcess Group PGID: 51240"
                    ),
                    Session::shell(
                        "c3",
                        "sleep 100",
                        "Command was cancelled by user before it could complete. There was no output before it was cancelled."
                    ),
                    Session::call(
                        "c4",
                        "run_shell_command",
                        json!({"command": "rm -rf /"}),
                        "error",
                        json!({"error": "Tool execution denied by policy."})
                    ),
                    Session::call(
                        "c5",
                        "run_shell_command",
                        json!({"command": "make"}),
                        "cancelled",
                        json!({"error": "User denied execution."})
                    ),
                    json!({"id": "c6", "name": "run_shell_command", "args": {"command": "npm run dev"}, "status": "executing"}),
                ]),
            ),
        ]);
        let calls = Session::calls(&trace);

        assert_eq!(calls.len(), 6);
        assert_eq!(
            calls[0].outcome,
            ToolOutcome::Failed {
                exit_code: Some(101)
            }
        );
        assert!(calls[0].error.is_some());
        assert_eq!(
            calls[0].args.command.as_deref(),
            Some("cargo clippy -- -D warnings")
        );
        assert_eq!(calls[1].outcome, ToolOutcome::Succeeded);
        assert_eq!(calls[1].action, ToolAction::Run);
        for call in &calls[2..] {
            assert_eq!(call.outcome, ToolOutcome::Interrupted, "{call:?}");
        }
        assert_eq!(trace.model.as_deref(), Some("gemini-3-pro-preview"));
    }

    #[test]
    fn commands_in_a_subdirectory_change_into_it_first() {
        let trace = Session::parse(&[
            Session::metadata(),
            Session::user("a1", "run the web tests"),
            Session::gemini(
                "b1",
                json!([
                    Session::call(
                        "c1",
                        "run_shell_command",
                        json!({"command": "npm test", "dir_path": "web"}),
                        "success",
                        json!({"output": "Output: ok"})
                    ),
                    Session::shell("c2", "git status", "Output: clean"),
                ]),
            ),
        ]);
        let calls = Session::calls(&trace);

        assert_eq!(calls[0].args.command.as_deref(), Some("cd web && npm test"));
        assert_eq!(calls[1].args.command.as_deref(), Some("git status"));
    }

    #[test]
    fn edits_count_lines_from_the_file_diff() {
        let diff = "Index: paginate.js\n===================================================================\n--- paginate.js\tCurrent\n+++ paginate.js\tProposed\n@@ -3,3 +3,3 @@\n export function paginate(items, page, perPage) {\n-  const end = start + perPage + 1;\n+  const end = start + perPage;\n";
        let trace = Session::parse(&[
            Session::metadata(),
            Session::user("a1", "fix paging"),
            Session::gemini(
                "b1",
                json!([
                    {"id": "e1", "name": "replace", "status": "success", "timestamp": "2026-10-09T10:13:11Z",
                     "args": {"file_path": "/home/dev/shop/src/paginate.js", "old_string": "+ 1;", "new_string": ";"},
                     "result": [{"functionResponse": {"id": "e1", "name": "replace", "response": {"output": "Successfully modified file"}}}],
                     "resultDisplay": {"fileDiff": diff, "fileName": "paginate.js", "filePath": "/home/dev/shop/src/paginate.js", "originalContent": "export function paginate(items, page, perPage) {\n", "newContent": "x"}},
                    {"id": "e2", "name": "write_file", "status": "success", "timestamp": "2026-10-09T10:13:12Z",
                     "args": {"file_path": "/home/dev/shop/test/paginate.test.js", "content": "a\nb\n"},
                     "result": [{"functionResponse": {"id": "e2", "name": "write_file", "response": {"output": "Created"}}}],
                     "resultDisplay": {"fileDiff": "@@ -0,0 +1,2 @@\n+a\n+b\n", "filePath": "/home/dev/shop/test/paginate.test.js", "originalContent": null, "newContent": "a\nb\n", "isNewFile": true}},
                    {"id": "e3", "name": "replace", "status": "error", "timestamp": "2026-10-09T10:13:13Z",
                     "args": {"file_path": "/home/dev/shop/src/x.js", "old_string": "a", "new_string": "b"},
                     "result": [{"functionResponse": {"id": "e3", "name": "replace", "response": {"error": "Failed to edit, could not find the string to replace."}}}]},
                ]),
            ),
        ]);
        let calls = Session::calls(&trace);

        assert_eq!(calls[0].action, ToolAction::Edit);
        assert_eq!(calls[0].args.path.as_deref(), Some("src/paginate.js"));
        let change = &calls[0].changes[0];
        assert_eq!(
            (change.lines_added, change.lines_removed, change.created),
            (1, 1, false)
        );
        assert_eq!(change.path, "src/paginate.js");
        let created = &calls[1].changes[0];
        assert_eq!((created.lines_added, created.created), (2, true));
        assert_eq!(calls[2].outcome, ToolOutcome::Failed { exit_code: None });
        assert!(calls[2].changes.is_empty());
    }

    #[test]
    fn re_appended_messages_count_once_with_their_last_state() {
        let pending = Session::gemini("b1", json!([]));
        let done = Session::gemini(
            "b1",
            json!([Session::shell("c1", "npm test", "Output: ok")]),
        );
        let trace = Session::parse(&[
            Session::metadata(),
            Session::user("a1", "run the tests"),
            json!({"$set": {"lastUpdated": "2026-10-09T10:12:31.501Z"}}),
            pending,
            done.clone(),
            done,
        ]);

        assert_eq!(Session::calls(&trace).len(), 1);
        assert_eq!(Session::prompts(&trace), ["run the tests"]);
    }

    #[test]
    fn rewinds_drop_the_rewound_messages() {
        let trace = Session::parse(&[
            Session::metadata(),
            Session::user("a1", "first try"),
            Session::gemini(
                "b1",
                json!([Session::shell("c1", "npm test", "Output: ok")]),
            ),
            json!({"$rewindTo": "a1"}),
            Session::user("a2", "second try"),
        ]);

        assert_eq!(Session::prompts(&trace), ["second try"]);
        assert!(Session::calls(&trace).is_empty());
    }

    #[test]
    fn compression_checkpoints_mark_a_compaction_and_keep_history() {
        let trace = Session::parse(&[
            Session::metadata(),
            Session::user("a1", "fix paging"),
            Session::gemini(
                "b1",
                json!([Session::shell("c1", "npm test", "Output: ok")]),
            ),
            json!({"$set": {"messages": [
                {"id": "s1", "timestamp": "2026-10-09T10:20:00Z", "type": "user", "content": [{"text": "<state_snapshot>\nfixed paging\n</state_snapshot>"}]},
                {"id": "s2", "timestamp": "2026-10-09T10:20:00Z", "type": "gemini", "content": "Got it. Thanks for the additional context!"},
                {"id": "b1", "timestamp": "2026-10-09T10:12:35Z", "type": "gemini", "content": "masked"}
            ]}}),
            Session::user("a2", "now the sorting"),
        ]);
        let kinds: Vec<&str> = trace
            .events
            .iter()
            .map(|event| match &event.kind {
                EventKind::Prompt { .. } => "prompt",
                EventKind::ToolCall(_) => "call",
                EventKind::Compaction { .. } => "compaction",
                _ => "other",
            })
            .collect();

        assert_eq!(kinds, ["prompt", "call", "compaction", "prompt"]);
    }

    #[test]
    fn legacy_single_object_sessions_parse() {
        let legacy = json!({
            "sessionId": "legacy-1", "projectHash": "3b1f", "startTime": "2026-05-01T09:00:00Z", "lastUpdated": "2026-05-01T09:05:00Z",
            "messages": [
                Session::user("a1", "fix paging"),
                Session::gemini("b1", json!([Session::shell("c1", "npm test", "Output: ok")]))
            ]
        });
        let text = serde_json::to_string_pretty(&legacy).expect("JSON");
        let trace = Transcript::parse(&text, Some(ROOT), &Redactor::with_home("/home/dev"))
            .expect("legacy parses");

        assert_eq!(trace.session.as_str(), "legacy-1");
        assert_eq!(Session::prompts(&trace), ["fix paging"]);
        assert_eq!(Session::calls(&trace).len(), 1);
    }

    #[test]
    fn malformed_lines_and_unknown_records_are_skipped() {
        let mut text = Session::text(&[Session::metadata(), Session::user("a1", "fix paging")]);
        text.push_str("not json\n{\"id\": 3}\n[1,2]\n{\"type\":\"info\",\"id\":\"i1\",\"content\":\"\"}\n{\"id\":\"b9\",\"type\":\"gemini\",\"toolCalls\":[{\"name\":5},{\"id\":\"z\"}]}\n");
        let trace = Transcript::parse(&text, Some(ROOT), &Redactor::with_home("/home/dev"))
            .expect("still parses");

        assert_eq!(Session::prompts(&trace), ["fix paging"]);
        assert!(Transcript::parse("garbage\n", None, &Redactor::with_home("/home/dev")).is_err());
    }

    #[test]
    fn subagent_transcripts_are_refused() {
        let text = Session::text(&[
            json!({"sessionId": "sub", "projectHash": "h", "kind": "subagent", "directories": ["/home/dev/shop"]}),
        ]);

        assert!(Transcript::parse(&text, Some(ROOT), &Redactor::with_home("/home/dev")).is_err());
        assert_eq!(Transcript::directories(&text), ["/home/dev/shop"]);
        assert_eq!(Transcript::session(&text).as_deref(), Some("sub"));
    }

    #[test]
    fn shell_results_parse_with_and_without_hook_context() {
        let failed = ShellResult::parse(
            "<untrusted_context>\nOutput: boom\nError: (none)\nExit Code: 2\nProcess Group PGID: 7\n</untrusted_context>\n\n<hook_context>&lt;m&gt;</hook_context>",
        );
        assert_eq!(failed.output, "boom\nError: (none)");
        assert_eq!(failed.exit_code, Some(2));
        assert!(failed.failed && !failed.interrupted);

        let passed = ShellResult::parse("Output: (empty)\nProcess Group PGID: 7");
        assert_eq!(passed.output, "");
        assert!(!passed.failed);

        let killed = ShellResult::parse("Output: x\nSignal: SIGKILL");
        assert!(killed.failed);
        assert_eq!(killed.exit_code, None);

        assert!(ShellResult::parse("Command was automatically cancelled because it exceeded the timeout of 5.0 minutes without output.").interrupted);
    }
}
