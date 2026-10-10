use std::{collections::HashMap, path::PathBuf};

use anyhow::Result;
use serde_json::Value;
use trodden_core::{
    Trace,
    trace::{FileChange, ToolAction, ToolArgs},
};
use trodden_redact::Redactor;

use crate::{
    ErrorSignature,
    builder::{Finish, Prompts, TraceBuilder},
    codex::PatchFile,
    command::Command,
    diff::Diff,
};

pub const HARNESS: &str = "copilot";

#[derive(Debug)]
pub struct Events;

impl Events {
    const PROMPTS: Prompts = Prompts {
        skipped: &["<system_reminder>", "<current_datetime>"],
        pasted: None,
    };

    const INJECTED_SOURCES: &[&str] = &[
        "skill",
        "agent",
        "system",
        "schedule",
        "command",
        "notification",
        "hook",
        "autopilot",
        "background",
    ];

    const SHELL_TOOLS: &[&str] = &["bash", "powershell", "shell", "local_shell"];

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "update_todo",
        "report_intent",
        "think",
        "ask_user",
        "read_bash",
        "write_bash",
        "stop_bash",
        "list_bash",
        "read_powershell",
        "write_powershell",
        "stop_powershell",
        "list_powershell",
        "exit_plan_mode",
        "fetch_copilot_cli_documentation",
        "skill",
    ];

    const MAX_SPLIT_LINES: usize = 64;

    pub fn parse(text: &str, redactor: &Redactor) -> Result<Trace> {
        let mut reader = Reader::new(redactor);
        for event in Self::events(text) {
            reader.push(event);
        }
        reader
            .builder
            .finish("no Copilot session found in events.jsonl")
    }

    pub fn working_directory(text: &str) -> Option<PathBuf> {
        Self::events(text)
            .into_iter()
            .filter(|event| event.kind == "session.start" || event.kind == "session.resume")
            .find_map(|event| {
                event
                    .data
                    .pointer("/context/cwd")
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
            })
    }

    pub fn failed(command: &str, output: &str, redactor: &Redactor) -> bool {
        if let Some(code) = Self::exit_marker(output) {
            return code != 0;
        }
        Command::normalize(command, "").action() == ToolAction::Run
            && ErrorSignature::of(output, redactor).is_some()
    }

    fn exit_marker(output: &str) -> Option<i32> {
        output.lines().rev().find_map(|line| {
            let line = line.trim();
            let inner = line.strip_prefix('<')?.strip_suffix('>')?;
            let (_, code) = inner.rsplit_once("exit code ")?;
            code.trim().parse().ok()
        })
    }

    fn events(text: &str) -> Vec<Event> {
        let mut events = Vec::new();
        let mut pending = String::new();
        let mut pending_lines = 0;
        for line in text.lines() {
            if !pending.is_empty() {
                let joined = format!("{pending}\\n{line}");
                if let Some(event) = Event::parse(&joined) {
                    events.push(event);
                    pending.clear();
                    continue;
                }
                pending = joined;
                pending_lines += 1;
                if line.starts_with("{\"") || pending_lines > Self::MAX_SPLIT_LINES {
                    pending.clear();
                } else {
                    continue;
                }
            }
            match Event::parse(line) {
                Some(event) => events.push(event),
                None if line.starts_with('{') => {
                    line.clone_into(&mut pending);
                    pending_lines = 0;
                }
                None => {}
            }
        }
        events
    }
}

#[derive(Debug)]
struct Event {
    kind: String,
    timestamp: Option<String>,
    subagent: bool,
    data: Value,
}

impl Event {
    fn parse(text: &str) -> Option<Self> {
        let Value::Object(mut fields) = serde_json::from_str(text).ok()? else {
            return None;
        };
        let Some(Value::String(kind)) = fields.remove("type") else {
            return None;
        };
        if fields.get("ephemeral").and_then(Value::as_bool) == Some(true) {
            return None;
        }
        let data = fields.remove("data").unwrap_or_default();
        let subagent = fields.get("agentId").is_some_and(|agent| !agent.is_null())
            || data
                .get("parentToolCallId")
                .is_some_and(|parent| !parent.is_null());
        Some(Self {
            kind,
            timestamp: fields
                .remove("timestamp")
                .and_then(|at| at.as_str().map(str::to_owned)),
            subagent,
            data,
        })
    }

    fn field<'v>(value: &'v Value, key: &str) -> Option<&'v str> {
        value.get(key).and_then(Value::as_str)
    }
}

struct Reader<'a> {
    builder: TraceBuilder<'a>,
    changes: HashMap<String, Vec<FileChange>>,
}

impl<'a> Reader<'a> {
    fn new(redactor: &'a Redactor) -> Self {
        Self {
            builder: TraceBuilder::new(HARNESS, redactor),
            changes: HashMap::new(),
        }
    }

    fn push(&mut self, event: Event) {
        if event.subagent {
            return;
        }
        if let Some(at) = &event.timestamp {
            self.builder.time(at);
        }
        let data = &event.data;
        match event.kind.as_str() {
            "session.start" => {
                if let Some(id) = Event::field(data, "sessionId") {
                    self.builder.session(id);
                }
                self.context(data.get("context"));
                if let Some(model) = Event::field(data, "selectedModel") {
                    self.builder.model(model);
                }
            }
            "session.resume" => self.context(data.get("context")),
            "session.context_changed" => self.context(Some(data)),
            "user.message" => self.prompt(data),
            "assistant.message" => {
                if let Some(model) = Event::field(data, "model") {
                    self.builder.model(model);
                }
                for request in data
                    .get("toolRequests")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    self.call(
                        Event::field(request, "toolCallId"),
                        Event::field(request, "name"),
                        request.get("arguments"),
                    );
                }
            }
            "tool.execution_start" => self.call(
                Event::field(data, "toolCallId"),
                Event::field(data, "toolName"),
                data.get("arguments"),
            ),
            "tool.execution_complete" => self.complete(data),
            "session.compaction_complete"
                if data.get("success").and_then(Value::as_bool) != Some(false) =>
            {
                let manual = Event::field(data, "trigger") == Some("manual");
                self.builder.compaction(!manual);
            }
            _ => {}
        }
    }

    fn context(&mut self, context: Option<&Value>) {
        if let Some(cwd) = context
            .and_then(|context| Event::field(context, "cwd"))
            .filter(|cwd| !cwd.is_empty())
        {
            self.builder.directory(cwd);
        }
    }

    fn prompt(&mut self, data: &Value) {
        if data.get("isAutopilotContinuation").and_then(Value::as_bool) == Some(true) {
            return;
        }
        let injected = Event::field(data, "source").is_some_and(|source| {
            Events::INJECTED_SOURCES
                .iter()
                .any(|prefix| source.starts_with(prefix))
        });
        if injected {
            return;
        }
        if let Some(text) = Event::field(data, "content") {
            self.builder.prompt(text, &Events::PROMPTS);
        }
    }

    fn call(&mut self, id: Option<&str>, name: Option<&str>, arguments: Option<&Value>) {
        let (Some(id), Some(name)) = (id, name) else {
            return;
        };
        if self.builder.is_pending(id) || Events::BOOKKEEPING_TOOLS.contains(&name) {
            return;
        }
        let args = match arguments {
            Some(Value::String(text)) => {
                serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
            }
            Some(value) => value.clone(),
            None => Value::Null,
        };
        let field = |key: &str| Event::field(&args, key);
        let builder = &self.builder;
        let mut tool_args = ToolArgs::default();
        let mut changes = Vec::new();
        let action = match name {
            _ if Events::SHELL_TOOLS.contains(&name) => {
                let (shell, action) = builder.shell(field("command").unwrap_or_default());
                tool_args = shell;
                action
            }
            "view" | "read" | "read_file" => {
                tool_args.path = field("path").map(|path| builder.relative(path));
                ToolAction::Read
            }
            "create" | "edit" | "write" | "str_replace" | "insert" | "str_replace_editor" => {
                let path = field("path").unwrap_or_default();
                let command = field("command").unwrap_or(name);
                if command == "view" {
                    tool_args.path = Some(builder.relative(path));
                    ToolAction::Read
                } else {
                    tool_args.path = Some(builder.relative(path));
                    let (created, diff) = match command {
                        "create" | "write" => {
                            (true, Diff::created(field("file_text").unwrap_or_default()))
                        }
                        "insert" => (false, Diff::created(field("new_str").unwrap_or_default())),
                        "edit" | "str_replace" => (
                            false,
                            Diff::replaced(
                                field("old_str").unwrap_or_default(),
                                field("new_str").unwrap_or_default(),
                            ),
                        ),
                        _ => (false, Diff::default()),
                    };
                    changes.push(builder.change(path, created, &diff, None));
                    ToolAction::Edit
                }
            }
            "apply_patch" => {
                let patch = field("input")
                    .or_else(|| field("patch"))
                    .or_else(|| args.as_str())
                    .unwrap_or_default();
                let files = PatchFile::parse(patch);
                tool_args.path = files.first().map(|file| builder.relative(&file.path));
                changes = files
                    .iter()
                    .map(|file| {
                        builder.change(&file.path, file.created, &PatchFile::diff(&file.body), None)
                    })
                    .collect();
                ToolAction::Edit
            }
            "grep" | "rg" | "glob" | "search" => {
                tool_args.pattern = field("pattern")
                    .or_else(|| field("query"))
                    .map(|pattern| builder.search_pattern(pattern, name == "glob"));
                tool_args.path = field("path").map(|path| builder.relative(path));
                ToolAction::Search
            }
            "web_fetch" | "fetch" => {
                tool_args.url = field("url").map(|url| builder.redact(url));
                ToolAction::Fetch
            }
            "web_search" => {
                tool_args.query = field("query").map(|query| builder.redact(query));
                ToolAction::Fetch
            }
            "task" | "agent" | "delegate" => ToolAction::Delegate,
            _ => ToolAction::Other,
        };
        if !changes.is_empty() {
            self.changes.insert(id.to_owned(), changes);
        }
        self.builder.call(Some(id), name, action, tool_args);
    }

    fn complete(&mut self, data: &Value) {
        let Some(id) = Event::field(data, "toolCallId") else {
            return;
        };
        if !self.builder.is_pending(id) {
            return;
        }
        let result = data.get("result").unwrap_or(&Value::Null);
        let error = data.get("error").unwrap_or(&Value::Null);
        let output = Event::field(result, "content")
            .or_else(|| Event::field(error, "message"))
            .unwrap_or_default()
            .to_owned();
        let exit_code = Self::exit_code(data, &output);
        let succeeded = data.get("success").and_then(Value::as_bool) != Some(false);
        let finish = if !succeeded && Self::declined(error) {
            Finish::Interrupted
        } else if !succeeded || exit_code.is_some_and(|code| code != 0) {
            Finish::Failed { exit_code }
        } else {
            Finish::Succeeded
        };
        let mut changes = self.changes.remove(id).unwrap_or_default();
        for edit in data
            .get("fileEdits")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(path) = Event::field(edit, "path") else {
                continue;
            };
            let relative = self.builder.relative(path);
            if !changes.iter().any(|change| change.path == relative) {
                let created = Event::field(edit, "kind") == Some("create");
                changes.push(self.builder.change(path, created, &Diff::default(), None));
            }
        }
        self.builder.resolve(id, finish, &output, None, changes);
    }

    fn exit_code(data: &Value, output: &str) -> Option<i32> {
        let structured = data
            .pointer("/shellExecution/exitCode")
            .and_then(Value::as_i64)
            .or_else(|| {
                data.pointer("/result/contents")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter(|block| Event::field(block, "type") == Some("shell_exit"))
                    .find_map(|block| block.get("exitCode").and_then(Value::as_i64))
            })
            .and_then(|code| i32::try_from(code).ok());
        structured.or_else(|| Events::exit_marker(output))
    }

    fn declined(error: &Value) -> bool {
        let code = Event::field(error, "code")
            .unwrap_or_default()
            .to_lowercase();
        let message = Event::field(error, "message")
            .unwrap_or_default()
            .to_lowercase();
        ["denied", "rejected", "cancel", "abort"]
            .iter()
            .any(|word| code.contains(word))
            || [
                "rejected",
                "denied by the user",
                "user denied",
                "cancelled",
                "canceled",
                "aborted",
            ]
            .iter()
            .any(|phrase| message.contains(phrase))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use trodden_core::trace::{EventKind, ToolCall, ToolOutcome};

    use super::*;

    const CWD: &str = "/home/dev/shop";
    const SESSION: &str = "3f2c9a4e-8b1d-4c6e-9a7f-1d2e3f4a5b6c";

    #[derive(Debug, Default)]
    struct Session {
        lines: Vec<String>,
    }

    impl Session {
        fn new() -> Self {
            let mut session = Self::default();
            session.event(
                "session.start",
                json!({"sessionId": SESSION, "version": 1, "producer": "copilot-agent",
                       "copilotVersion": "1.0.94", "startTime": "2026-10-09T10:00:00.000Z",
                       "selectedModel": "claude-sonnet-4.5",
                       "context": {"cwd": CWD, "gitRoot": CWD, "branch": "main"}}),
            );
            session
        }

        fn event(&mut self, kind: &str, data: Value) -> &mut Self {
            let line = json!({
                "id": format!("00000000-0000-4000-8000-{:012}", self.lines.len()),
                "timestamp": format!("2026-10-09T10:00:{:02}.000Z", self.lines.len() % 60),
                "parentId": null,
                "type": kind,
                "data": data,
            });
            self.lines.push(line.to_string());
            self
        }

        fn prompt(&mut self, text: &str) -> &mut Self {
            self.event("user.message", json!({"content": text,
                "transformedContent": format!("<current_datetime>2026-10-09</current_datetime>\n{text}"),
                "attachments": [], "interactionId": "i1"}))
        }

        fn tool(&mut self, id: &str, name: &str, arguments: Value) -> &mut Self {
            self.event("assistant.message", json!({"messageId": "m1", "content": "", "model": "claude-sonnet-4.5",
                    "toolRequests": [{"toolCallId": id, "name": name, "arguments": arguments, "type": "function"}]}))
                .event("tool.execution_start", json!({"toolCallId": id, "toolName": name, "arguments": arguments}))
        }

        fn done(&mut self, id: &str, data: Value) -> &mut Self {
            let mut data = data;
            data["toolCallId"] = json!(id);
            self.event("tool.execution_complete", data)
        }

        fn text(&self) -> String {
            self.lines.iter().map(|line| format!("{line}\n")).collect()
        }

        fn trace(&self) -> Trace {
            Events::parse(&self.text(), &Redactor::with_home("/home/dev")).expect("events parse")
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
    fn typed_prompts_are_kept_and_injected_ones_skipped() {
        let mut session = Session::new();
        session
            .prompt("Page 2 repeats the last product. Fix it.\nDetails follow")
            .event("user.message", json!({"content": "Use the pdf skill", "source": "skill-pdf"}))
            .event("user.message", json!({"content": "Keep going", "isAutopilotContinuation": true}))
            .event("user.message", json!({"content": "Trodden: the procedure recalled for this task is checked with `npm test`"}))
            .event("user.message", json!({"content": "ephemeral"}));
        let last = session.lines.last_mut().expect("events were added");
        *last = last.replacen("\"parentId\"", "\"ephemeral\":true,\"parentId\"", 1);

        let trace = session.trace();

        assert_eq!(trace.session.as_str(), SESSION);
        assert_eq!(trace.harness.as_str(), "copilot");
        assert_eq!(trace.cwd, "~/shop");
        assert_eq!(trace.model.as_deref(), Some("claude-sonnet-4.5"));
        assert_eq!(
            session.prompts(),
            ["Page 2 repeats the last product. Fix it."]
        );
    }

    #[test]
    fn shell_calls_take_their_exit_code_from_the_completion() {
        let mut session = Session::new();
        session
            .prompt("Fix the paging bug")
            .tool("call_1", "bash", json!({"command": "npm test", "description": "Run tests"}))
            .done("call_1", json!({"success": true, "result": {"content": "Error: Cannot find module 'left-pad'\n<shellId: 1 completed with exit code 1>"},
                "shellExecution": {"exitCode": 1}}))
            .tool("call_2", "bash", json!({"command": "npm run lint"}))
            .done("call_2", json!({"success": true, "result": {"content": "lint failed\n<exited with exit code 2>"}}))
            .tool("call_3", "bash", json!({"command": "npm run build"}))
            .done("call_3", json!({"success": true, "result": {"content": "built",
                "contents": [{"type": "shell_exit", "shellId": "3", "exitCode": 0}]}}))
            .tool("call_4", "bash", json!({"command": "rm -rf dist"}))
            .done("call_4", json!({"success": false, "error": {"message": "The user rejected this tool call", "code": "rejected"}}))
            .tool("call_5", "bash", json!({"command": "npm start"}));

        let calls = session.calls();
        let outcomes: Vec<&ToolOutcome> = calls.iter().map(|call| &call.outcome).collect();

        assert_eq!(calls.len(), 5, "requests and starts are one call");
        assert_eq!(calls[0].args.command.as_deref(), Some("npm test"));
        assert_eq!(calls[0].action, ToolAction::Run);
        assert_eq!(
            outcomes,
            [
                &ToolOutcome::Failed { exit_code: Some(1) },
                &ToolOutcome::Failed { exit_code: Some(2) },
                &ToolOutcome::Succeeded,
                &ToolOutcome::Interrupted,
                &ToolOutcome::Interrupted,
            ]
        );
        assert!(
            calls[0]
                .error
                .as_deref()
                .is_some_and(|error| error.contains("cannot find module"))
        );
    }

    #[test]
    fn edits_record_their_changes() {
        let mut session = Session::new();
        session
            .prompt("Fix the paging bug")
            .tool("call_view", "view", json!({"path": "/home/dev/shop/src/paginate.js"}))
            .done("call_view", json!({"success": true, "result": {"content": "1. export function paginate"}}))
            .tool("call_edit", "edit", json!({"path": "/home/dev/shop/src/paginate.js",
                "old_str": "export function paginate(items) {\n  return items.slice(start, end + 1);\n}",
                "new_str": "export function paginate(items) {\n  return items.slice(start, end);\n}"}))
            .done("call_edit", json!({"success": true, "result": {"content": "File updated"},
                "fileEdits": [{"path": "/home/dev/shop/src/paginate.js", "kind": "edit"}]}))
            .tool("call_create", "create", json!({"path": "/home/dev/shop/test/paging.test.js", "file_text": "a\nb\nc\n"}))
            .done("call_create", json!({"success": true, "result": {"content": "Created"}}))
            .tool("call_patch", "apply_patch", json!({"input": "*** Begin Patch\n*** Update File: src/cart.js\n@@ function total\n-  sum\n+  sum + tax\n*** End Patch"}))
            .done("call_patch", json!({"success": true, "result": {"content": "Done"},
                "fileEdits": [{"path": "/home/dev/shop/src/cart.js", "kind": "edit"}, {"path": "/home/dev/shop/src/tax.js", "kind": "create"}]}))
            .tool("call_bad", "edit", json!({"path": "/home/dev/shop/src/x.js", "old_str": "a", "new_str": "b"}))
            .done("call_bad", json!({"success": false, "error": {"message": "No match found for old_str"}}));

        let calls = session.calls();

        assert_eq!(calls[0].action, ToolAction::Read);
        assert_eq!(calls[0].args.path.as_deref(), Some("src/paginate.js"));
        let change = &calls[1].changes[0];
        assert_eq!(
            (
                change.path.as_str(),
                change.lines_added,
                change.lines_removed
            ),
            ("src/paginate.js", 1, 1)
        );
        assert_eq!(calls[1].changes.len(), 1);
        assert!(calls[2].changes[0].created);
        assert_eq!(calls[2].changes[0].lines_added, 3);
        let patched: Vec<(&str, bool, u32)> = calls[3]
            .changes
            .iter()
            .map(|change| (change.path.as_str(), change.created, change.lines_added))
            .collect();
        assert_eq!(
            patched,
            [("src/cart.js", false, 1), ("src/tax.js", true, 0)]
        );
        assert_eq!(calls[3].changes[0].symbols, ["total"]);
        assert_eq!(calls[4].outcome, ToolOutcome::Failed { exit_code: None });
        assert!(calls[4].changes.is_empty());
    }

    #[test]
    fn bookkeeping_and_subagents_stay_out_of_the_trace() {
        let mut session = Session::new();
        session
            .prompt("Fix the paging bug")
            .tool("call_todo", "update_todo", json!({"todos": "- [ ] fix"}))
            .done("call_todo", json!({"success": true}))
            .tool(
                "call_task",
                "task",
                json!({"agent_type": "explore", "prompt": "find paging code"}),
            )
            .event(
                "tool.execution_start",
                json!({"toolCallId": "sub_1", "toolName": "bash",
                "arguments": {"command": "rg paginate"}, "parentToolCallId": "call_task"}),
            )
            .done(
                "call_task",
                json!({"success": true, "result": {"content": "src/paginate.js"}}),
            );
        let line = json!({"id": "x", "timestamp": "2026-10-09T10:00:59.000Z", "parentId": null,
            "agentId": "explore-1", "type": "user.message", "data": {"content": "find paging code"}});
        session.lines.push(line.to_string());

        let calls = session.calls();

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].action, ToolAction::Delegate);
        assert_eq!(session.prompts().len(), 1);
    }

    #[test]
    fn compactions_are_marked() {
        let mut session = Session::new();
        session
            .prompt("Fix the paging bug")
            .event(
                "session.compaction_complete",
                json!({"success": true, "trigger": "threshold"}),
            )
            .event(
                "session.compaction_complete",
                json!({"success": true, "trigger": "manual"}),
            )
            .event(
                "session.compaction_complete",
                json!({"success": false, "error": "rate limited"}),
            );

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
    fn split_and_broken_lines_are_tolerated() {
        let mut session = Session::new();
        session.prompt("Fix the paging bug");
        let complete = json!({"id": "y", "timestamp": "2026-10-09T10:00:30.000Z", "parentId": null,
            "type": "tool.execution_complete",
            "data": {"toolCallId": "call_1", "success": true, "result": {"content": "line one\\nline two"}}})
        .to_string()
        .replace("line one\\\\nline two", "line one\nline two");
        let text = format!(
            "{}{}\n{}\n{}\nnot json\n{{\"id\":\"z\",\"type\":\"user.mess\n",
            session.text(),
            json!({"id": "x", "timestamp": "2026-10-09T10:00:29.000Z", "parentId": null, "type": "tool.execution_start",
                   "data": {"toolCallId": "call_1", "toolName": "bash", "arguments": {"command": "npm test"}}}),
            complete,
            json!({"id": "w", "timestamp": "2026-10-09T10:00:31.000Z", "parentId": null, "type": "user.message",
                   "data": {"content": "Now the cart"}}),
        );

        let trace = Events::parse(&text, &Redactor::with_home("/home/dev")).expect("events parse");
        let kinds: Vec<String> = trace
            .events
            .iter()
            .map(|event| match &event.kind {
                EventKind::Prompt { summary } => summary.clone(),
                EventKind::ToolCall(call) => format!("{:?}", call.outcome),
                other => format!("{other:?}"),
            })
            .collect();

        assert_eq!(kinds, ["Fix the paging bug", "Succeeded", "Now the cart"]);
        assert!(Events::parse("", &Redactor::with_home("/home/dev")).is_err());
    }

    #[test]
    fn working_directory_comes_from_the_session_start() {
        assert_eq!(
            Events::working_directory(&Session::new().text()),
            Some(PathBuf::from(CWD))
        );
    }

    #[test]
    fn hook_outputs_reveal_failures() {
        let redactor = Redactor::with_home("/home/dev");

        assert!(Events::failed(
            "npm test",
            "all good\n<exited with exit code 1>",
            &redactor
        ));
        assert!(!Events::failed(
            "npm test",
            "Error: boom\n<shellId: 2 completed with exit code 0>",
            &redactor
        ));
        assert!(Events::failed(
            "npm test",
            "Error: Cannot find module 'left-pad'\n",
            &redactor
        ));
        assert!(!Events::failed(
            "cat notes.txt",
            "Error: Cannot find module 'left-pad'\n",
            &redactor
        ));
    }
}
