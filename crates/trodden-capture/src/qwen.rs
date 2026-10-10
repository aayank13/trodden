use std::collections::{HashMap, HashSet};

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

pub const HARNESS: &str = "qwen";

#[derive(Debug)]
pub struct Transcript;

impl Transcript {
    const PROMPTS: Prompts = Prompts {
        skipped: &[
            "<qwen:user-prompt-submit-context>",
            "<state_snapshot>",
            "<system-reminder>",
        ],
        pasted: None,
    };

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "todo_write",
        "save_memory",
        "manage_memory",
        "search_memory",
        "skill",
        "exit_plan_mode",
        "enter_plan_mode",
        "ask_user_question",
        "cron_create",
        "cron_list",
        "cron_delete",
        "loop_wakeup",
        "list_agents",
        "task_stop",
        "task_create",
        "task_update",
        "task_list",
        "send_message",
        "structured_output",
        "tool_search",
        "get_goal",
        "update_goal",
        "monitor",
    ];

    pub fn parse(text: &str, redactor: &Redactor) -> Result<Trace> {
        let chain = Chain::new(text);
        if chain.managed {
            bail!("this Qwen Code session uses the managed engine, whose log is not a transcript");
        }
        let mut reader = Reader {
            builder: TraceBuilder::new(HARNESS, redactor),
        };
        for record in chain.live() {
            reader.push(record);
        }
        reader
            .builder
            .finish("no Qwen Code session found in transcript")
    }

    pub fn working_directory(text: &str) -> Option<String> {
        Chain::new(text)
            .live()
            .into_iter()
            .filter(|record| !Reader::is_sidechain(record))
            .find_map(|record| record.get("cwd").and_then(Value::as_str))
            .filter(|cwd| !cwd.is_empty())
            .map(str::to_owned)
    }
}

#[derive(Debug, Default)]
struct Chain {
    order: Vec<String>,
    records: HashMap<String, Value>,
    managed: bool,
}

impl Chain {
    fn new(text: &str) -> Self {
        let mut chain = Self::default();
        for line in text.lines() {
            let Ok(record @ Value::Object(_)) = serde_json::from_str::<Value>(line) else {
                continue;
            };
            if record
                .get("subtype")
                .and_then(Value::as_str)
                .is_some_and(|subtype| subtype.starts_with("managed_session_"))
            {
                chain.managed = true;
            }
            let Some(uuid) = record
                .get("uuid")
                .and_then(Value::as_str)
                .map(str::to_owned)
            else {
                continue;
            };
            match chain.records.get_mut(&uuid) {
                Some(first) => Self::merge(first, record),
                None => {
                    chain.order.push(uuid.clone());
                    chain.records.insert(uuid, record);
                }
            }
        }
        chain
    }

    fn merge(first: &mut Value, later: Value) {
        let Value::Object(later) = later else { return };
        for (key, value) in later {
            if key == "message" {
                let parts = value
                    .get("parts")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                match first
                    .get_mut("message")
                    .and_then(|message| message.get_mut("parts"))
                    .and_then(Value::as_array_mut)
                {
                    Some(existing) => existing.extend(parts),
                    None => {
                        first["message"] = value;
                    }
                }
            } else if first.get(&key).is_none_or(Value::is_null) {
                first[key.as_str()] = value;
            }
        }
    }

    fn live(&self) -> Vec<&Value> {
        let Some(mut current) = self.order.last() else {
            return Vec::new();
        };
        let mut live = HashSet::new();
        while live.insert(current.as_str()) {
            let Some(parent) = self
                .records
                .get(current)
                .and_then(|record| record.get("parentUuid"))
                .and_then(Value::as_str)
                .and_then(|parent| self.records.get_key_value(parent))
                .map(|(uuid, _)| uuid)
            else {
                break;
            };
            current = parent;
        }
        self.order
            .iter()
            .filter(|uuid| live.contains(uuid.as_str()))
            .filter_map(|uuid| self.records.get(uuid))
            .collect()
    }
}

struct Reader<'a> {
    builder: TraceBuilder<'a>,
}

impl Reader<'_> {
    fn is_sidechain(record: &Value) -> bool {
        record.get("isSidechain").and_then(Value::as_bool) == Some(true)
            || record.get("agentId").is_some_and(|id| !id.is_null())
    }

    fn push(&mut self, record: &Value) {
        if Self::is_sidechain(record) {
            return;
        }
        let field = |key: &str| record.get(key).and_then(Value::as_str);
        if let Some(session) = field("sessionId") {
            self.builder.session(session);
        }
        if let Some(cwd) = field("cwd") {
            self.builder.directory(cwd);
        }
        if let Some(at) = field("timestamp") {
            self.builder.time(at);
        }
        let subtype = field("subtype");
        match (field("type"), subtype) {
            (Some("user"), None) => self.push_user(record),
            (Some("assistant"), _) => self.push_assistant(record),
            (Some("tool_result"), None) => self.push_result(record),
            (Some("system"), Some("chat_compression")) => {
                let manual = record
                    .get("systemPayload")
                    .and_then(|payload| payload.get("trigger"))
                    .and_then(Value::as_str)
                    == Some("manual");
                self.builder.compaction(!manual);
            }
            _ => {}
        }
    }

    fn parts(record: &Value) -> &[Value] {
        record
            .get("message")
            .and_then(|message| message.get("parts"))
            .and_then(Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    fn push_user(&mut self, record: &Value) {
        let provenance = record.get("provenance").and_then(Value::as_str);
        if provenance.is_some_and(|provenance| provenance != "real_user") {
            return;
        }
        let shown = record
            .get("systemPayload")
            .and_then(|payload| payload.get("displayText"))
            .and_then(Value::as_str);
        let text = shown.or_else(|| {
            Self::parts(record)
                .iter()
                .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .find(|text| !text.trim_start().starts_with("<qwen:"))
        });
        if let Some(text) = text {
            self.builder.prompt(text, &Transcript::PROMPTS);
        }
    }

    fn push_assistant(&mut self, record: &Value) {
        if let Some(model) = record.get("model").and_then(Value::as_str) {
            self.builder.model(model);
        }
        for part in Self::parts(record) {
            let Some(call) = part.get("functionCall") else {
                continue;
            };
            let (Some(id), Some(name)) = (
                call.get("id").and_then(Value::as_str),
                call.get("name").and_then(Value::as_str),
            ) else {
                continue;
            };
            if Transcript::BOOKKEEPING_TOOLS.contains(&name) || self.builder.is_pending(id) {
                continue;
            }
            let args = call.get("args").cloned().unwrap_or_default();
            if let Some((action, tool_args)) = self.tool_call(name, &args) {
                self.builder.call(Some(id), name, action, tool_args);
            }
        }
    }

    fn tool_call(&mut self, name: &str, args: &Value) -> Option<(ToolAction, ToolArgs)> {
        let field = |key: &str| args.get(key).and_then(Value::as_str);
        let builder = &self.builder;
        let mut tool_args = ToolArgs::default();
        let action = match name {
            "run_shell_command" => {
                let command = field("command").unwrap_or_default();
                let (shell, action) = match field("directory").filter(|dir| !dir.is_empty()) {
                    Some(dir) => self.shell_in(dir, command),
                    None => self.builder.shell(command),
                };
                tool_args = shell;
                action
            }
            "edit" | "replace" | "write_file" | "notebook_edit" => {
                tool_args.path = field("file_path")
                    .or_else(|| field("notebook_path"))
                    .map(|p| builder.relative(p));
                ToolAction::Edit
            }
            "read_file" | "list_directory" | "zoom_image" => {
                tool_args.path = field("file_path")
                    .or_else(|| field("path"))
                    .or_else(|| field("absolute_path"))
                    .map(|p| builder.relative(p));
                ToolAction::Read
            }
            "grep_search" | "search_file_content" | "glob" => {
                tool_args.pattern =
                    field("pattern").map(|p| builder.search_pattern(p, name == "glob"));
                tool_args.path = field("path").map(|p| builder.relative(p));
                ToolAction::Search
            }
            "web_fetch" => {
                tool_args.url = field("url").map(|u| builder.redact(u));
                ToolAction::Fetch
            }
            "web_search" => {
                tool_args.query = field("query").map(|q| builder.redact(q));
                ToolAction::Fetch
            }
            "agent" | "task" => ToolAction::Delegate,
            _ => ToolAction::Other,
        };
        Some((action, tool_args))
    }

    fn shell_in(&mut self, dir: &str, command: &str) -> (ToolArgs, ToolAction) {
        let root = self.builder.working_directory().to_owned();
        if root.is_empty() {
            return self.builder.shell(command);
        }
        let here = if TraceBuilder::is_absolute(dir) {
            dir.to_owned()
        } else {
            format!("{}/{}", root.trim_end_matches(['/', '\\']), dir)
        };
        self.builder.directory(&here);
        let shell = self.builder.shell(command);
        self.builder.directory(&root);
        shell
    }

    fn push_result(&mut self, record: &Value) {
        let result = record.get("toolCallResult");
        for part in Self::parts(record) {
            let Some(response) = part.get("functionResponse") else {
                continue;
            };
            let Some(id) = response
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| result.and_then(|r| r.get("callId")).and_then(Value::as_str))
            else {
                continue;
            };
            if !self.builder.is_pending(id) {
                continue;
            }
            self.resolve(id, response.get("response"), result);
        }
    }

    fn resolve(&mut self, id: &str, response: Option<&Value>, result: Option<&Value>) {
        let field = |key: &str| result.and_then(|r| r.get(key)).and_then(Value::as_str);
        let display = result.and_then(|r| r.get("resultDisplay"));
        let shell =
            display.filter(|d| d.get("type").and_then(Value::as_str) == Some("shell_result"));
        let error_text = response.and_then(|r| r.get("error")).map(Self::string);
        let output_text = response.and_then(|r| r.get("output")).map(Self::string);
        let status = field("status").unwrap_or(if error_text.is_some() {
            "error"
        } else {
            "success"
        });
        let denied = field("errorType")
            .is_some_and(|kind| kind == "execution_denied" || kind == "permission_denied");
        let exit_code = shell
            .and_then(|s| s.get("exitCode"))
            .and_then(Value::as_i64)
            .and_then(|code| i32::try_from(code).ok());
        let outcome = shell.and_then(|s| s.get("outcome")).and_then(Value::as_str);
        let finish = match status {
            "cancelled" => Finish::Interrupted,
            _ if denied || outcome == Some("cancelled") => Finish::Interrupted,
            "error" => Finish::Failed { exit_code },
            _ => Finish::Succeeded,
        };
        let output = shell
            .and_then(|s| s.get("output"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or(error_text)
            .or(output_text)
            .unwrap_or_default();
        let changes = if finish == Finish::Succeeded {
            display
                .and_then(|d| self.file_change(d))
                .into_iter()
                .collect()
        } else {
            Vec::new()
        };
        self.builder.resolve(id, finish, &output, None, changes);
    }

    fn string(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            other => other
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| other.to_string(), str::to_owned),
        }
    }

    fn file_change(&self, display: &Value) -> Option<FileChange> {
        let diff = display.get("fileDiff").and_then(Value::as_str)?;
        let path = display
            .get("filePath")
            .and_then(Value::as_str)
            .or_else(|| display.get("fileName").and_then(Value::as_str))?;
        let original = display.get("originalContent").and_then(Value::as_str);
        let created = display.get("originalContent").is_some_and(Value::is_null)
            || display.get("isNewFile").and_then(Value::as_bool) == Some(true);
        Some(
            self.builder
                .change(path, created, &Diff::unified(diff), original),
        )
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use trodden_core::trace::{EventKind, ToolCall, ToolOutcome};

    use super::*;

    #[derive(Debug, Default)]
    struct Session {
        lines: Vec<Value>,
    }

    impl Session {
        fn record(&mut self, kind: &str, extra: Value) -> &mut Self {
            let uuid = format!("r{}", self.lines.len() + 1);
            let parent = (!self.lines.is_empty()).then(|| format!("r{}", self.lines.len()));
            let mut record = json!({"uuid": uuid, "parentUuid": parent, "sessionId": "6f1c2a9e", "timestamp": "2026-10-09T10:12:03.600Z",
                                    "type": kind, "cwd": "/home/dev/shop", "version": "0.25.0"});
            record
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            self.lines.push(record);
            self
        }

        fn user(&mut self, text: &str) -> &mut Self {
            self.record("user", json!({"provenance": "real_user", "message": {"role": "user", "parts": [{"text": text}]}}))
        }

        fn calls(&mut self, calls: Value) -> &mut Self {
            let parts: Vec<Value> = calls
                .as_array()
                .expect("array")
                .iter()
                .map(|call| json!({"functionCall": call}))
                .collect();
            self.record("assistant", json!({"provenance": "assistant_output", "model": "qwen3-coder-plus", "message": {"role": "model", "parts": parts}}))
        }

        fn result(&mut self, id: &str, name: &str, response: Value, result: Value) -> &mut Self {
            self.record("tool_result", json!({"provenance": "tool_result",
                "message": {"role": "user", "parts": [{"functionResponse": {"id": id, "name": name, "response": response}}]},
                "toolCallResult": result}))
        }

        fn shell(&mut self, id: &str, exit_code: i64, output: &str) -> &mut Self {
            let status = if exit_code == 0 { "success" } else { "error" };
            let response = if exit_code == 0 {
                json!({"output": format!("Command: x\nOutput: {output}")})
            } else {
                json!({"error": format!("Command: x\nOutput: {output}\nExit Code: {exit_code}")})
            };
            self.result(id, "run_shell_command", response, json!({"callId": id, "status": status,
                "resultDisplay": {"type": "shell_result", "version": 1, "output": output, "directory": "/home/dev/shop", "exitCode": exit_code, "outcome": if exit_code == 0 { "completed" } else { "failed" }}}))
        }

        fn text(&self) -> String {
            self.lines.iter().map(|line| format!("{line}\n")).collect()
        }

        fn trace(&self) -> Trace {
            Transcript::parse(&self.text(), &Redactor::with_home("/home/dev"))
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

        fn tool_calls(trace: &Trace) -> Vec<ToolCall> {
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
    fn only_typed_prompts_are_kept() {
        let mut session = Session::default();
        session
            .record("user", json!({"provenance": "real_user",
                "message": {"role": "user", "parts": [{"text": "fix the migration test"}, {"text": "<qwen:user-prompt-submit-context>\n&lt;trodden-memory&gt;\n</qwen:user-prompt-submit-context>"}]},
                "systemPayload": {"displayText": "fix the migration test", "hookContext": "&lt;trodden-memory&gt;"}}))
            .record("user", json!({"subtype": "notification", "provenance": "system", "message": {"role": "user", "parts": [{"text": "Background task done"}]}}))
            .record("user", json!({"subtype": "cron", "provenance": "system", "message": {"role": "user", "parts": [{"text": "check CI"}]}}))
            .record("user", json!({"provenance": "goal_runtime", "message": {"role": "user", "parts": [{"text": "continue the goal"}]}}))
            .record("system", json!({"subtype": "slash_command", "systemPayload": {"phase": "invocation", "rawCommand": "/compress"}}))
            .user("Trodden: the procedure recalled for this task is checked with `npm test`, which has not run since your last change.")
            .user("now the sorting bug");
        let trace = session.trace();

        assert_eq!(
            Session::prompts(&trace),
            ["fix the migration test", "now the sorting bug"]
        );
        assert_eq!(trace.session.as_str(), "6f1c2a9e");
        assert_eq!(trace.cwd, "~/shop");
        assert_eq!(trace.harness.as_str(), HARNESS);
    }

    #[test]
    fn shell_outcomes_use_the_recorded_exit_code() {
        let mut session = Session::default();
        session
            .user("fix the db tests")
            .calls(json!([
                {"id": "call_1", "name": "run_shell_command", "args": {"command": "cargo test -p db", "is_background": false}},
                {"id": "call_2", "name": "run_shell_command", "args": {"command": "cargo test -p db", "directory": "crates/db"}},
                {"id": "call_3", "name": "run_shell_command", "args": {"command": "sleep 99"}},
                {"id": "call_4", "name": "run_shell_command", "args": {"command": "rm -rf build"}},
                {"id": "call_5", "name": "run_shell_command", "args": {"command": "npm run dev"}}
            ]))
            .shell("call_1", 101, "error[E0432]: unresolved import `sqlx::migrate`")
            .shell("call_2", 0, "test result: ok. 4 passed; 0 failed")
            .result("call_3", "run_shell_command", json!({"error": "cancelled"}), json!({"callId": "call_3", "status": "cancelled"}))
            .result("call_4", "run_shell_command", json!({"error": "denied"}), json!({"callId": "call_4", "status": "error", "errorType": "execution_denied"}));
        let calls = Session::tool_calls(&session.trace());

        assert_eq!(
            calls[0].outcome,
            ToolOutcome::Failed {
                exit_code: Some(101)
            }
        );
        assert!(
            calls[0]
                .error
                .as_deref()
                .is_some_and(|e| e.contains("sqlx")),
            "{:?}",
            calls[0].error
        );
        assert_eq!(calls[1].outcome, ToolOutcome::Succeeded);
        assert_eq!(
            calls[1].args.command.as_deref(),
            Some("cd crates/db && cargo test -p db")
        );
        assert_eq!(calls[2].outcome, ToolOutcome::Interrupted);
        assert_eq!(calls[3].outcome, ToolOutcome::Interrupted);
        assert_eq!(calls[4].outcome, ToolOutcome::Interrupted);
    }

    #[test]
    fn edits_count_lines_and_skip_failed_ones() {
        let mut session = Session::default();
        session
            .user("fix paging")
            .calls(json!([
                {"id": "e1", "name": "edit", "args": {"file_path": "/home/dev/shop/src/paginate.js", "old_string": "+ 1;", "new_string": ";"}},
                {"id": "e2", "name": "write_file", "args": {"file_path": "/home/dev/shop/test/p.test.js", "content": "a\nb\n"}},
                {"id": "e3", "name": "edit", "args": {"file_path": "/home/dev/shop/src/x.js", "old_string": "q", "new_string": "r"}}
            ]))
            .result("e1", "edit", json!({"output": "ok"}), json!({"callId": "e1", "status": "success",
                "resultDisplay": {"fileDiff": "--- paginate.js\n+++ paginate.js\n@@ -3,2 +3,2 @@\n-  const end = start + perPage + 1;\n+  const end = start + perPage;\n", "fileName": "paginate.js", "filePath": "/home/dev/shop/src/paginate.js", "originalContent": "x", "newContent": "y"}}))
            .result("e2", "write_file", json!({"output": "ok"}), json!({"callId": "e2", "status": "success",
                "resultDisplay": {"fileDiff": "@@ -0,0 +1,2 @@\n+a\n+b\n", "fileName": "p.test.js", "filePath": "/home/dev/shop/test/p.test.js", "originalContent": null, "newContent": "a\nb\n"}}))
            .result("e3", "edit", json!({"error": "could not find the string"}), json!({"callId": "e3", "status": "error"}));
        let calls = Session::tool_calls(&session.trace());

        assert_eq!(calls[0].args.path.as_deref(), Some("src/paginate.js"));
        let change = &calls[0].changes[0];
        assert_eq!(
            (
                change.path.as_str(),
                change.lines_added,
                change.lines_removed,
                change.created
            ),
            ("src/paginate.js", 1, 1, false)
        );
        assert_eq!(
            (calls[1].changes[0].lines_added, calls[1].changes[0].created),
            (2, true)
        );
        assert_eq!(calls[2].outcome, ToolOutcome::Failed { exit_code: None });
        assert!(calls[2].changes.is_empty());
    }

    #[test]
    fn rewound_branches_and_sidechains_are_left_out() {
        let mut session = Session::default();
        session.user("first attempt").calls(
            json!([{"id": "c1", "name": "run_shell_command", "args": {"command": "npm test"}}]),
        );
        session.shell("c1", 0, "ok");
        session
            .record("assistant", json!({"isSidechain": true, "agentId": "a1", "message": {"role": "model", "parts": [{"functionCall": {"id": "s1", "name": "run_shell_command", "args": {"command": "ls"}}}]}}));
        session.lines.push(json!({"uuid": "n1", "parentUuid": null, "sessionId": "6f1c2a9e", "timestamp": "2026-10-09T10:20:00Z", "type": "user", "provenance": "real_user", "cwd": "/home/dev/shop", "message": {"role": "user", "parts": [{"text": "second attempt"}]}}));
        session.lines.push(json!({"uuid": "n2", "parentUuid": "n1", "sessionId": "6f1c2a9e", "timestamp": "2026-10-09T10:20:01Z", "type": "assistant", "cwd": "/home/dev/shop", "message": {"role": "model", "parts": [{"text": "On it."}]}}));
        let trace = session.trace();

        assert_eq!(Session::prompts(&trace), ["second attempt"]);
        assert!(Session::tool_calls(&trace).is_empty());
    }

    #[test]
    fn fragments_of_one_record_merge() {
        let mut session = Session::default();
        session.user("run tests").calls(
            json!([{"id": "c1", "name": "run_shell_command", "args": {"command": "npm test"}}]),
        );
        let mut fragment = session.lines.last().expect("record").clone();
        fragment["message"]["parts"] = json!([{"functionCall": {"id": "c2", "name": "run_shell_command", "args": {"command": "npm run lint"}}}]);
        session.lines.push(fragment);
        let trace = session.trace();

        assert_eq!(Session::tool_calls(&trace).len(), 2);
    }

    #[test]
    fn compression_and_malformed_lines() {
        let mut session = Session::default();
        session
            .user("fix paging")
            .record("system", json!({"subtype": "chat_compression", "provenance": "system", "systemPayload": {"info": {}, "compressedHistory": [{"role": "user", "parts": [{"text": "<state_snapshot>"}]}]}}));
        let mut text = session.text();
        text.insert_str(0, "garbage\n{\"uuid\": 5}\n");
        let trace = Transcript::parse(&text, &Redactor::with_home("/home/dev")).expect("parses");

        assert!(
            trace
                .events
                .iter()
                .any(|event| matches!(event.kind, EventKind::Compaction { automatic: true }))
        );
        assert_eq!(
            Transcript::working_directory(&text).as_deref(),
            Some("/home/dev/shop")
        );
        assert!(Transcript::parse("nothing\n", &Redactor::with_home("/home/dev")).is_err());
    }

    #[test]
    fn managed_engine_logs_are_refused() {
        let mut session = Session::default();
        session.record("system", json!({"subtype": "managed_session_header_v1"}));

        assert!(Transcript::parse(&session.text(), &Redactor::with_home("/home/dev")).is_err());
    }
}
