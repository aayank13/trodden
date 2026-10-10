use anyhow::Result;
use jiff::Timestamp;
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

pub const HARNESS: &str = "droid";

#[derive(Debug)]
pub struct Transcript;

impl Transcript {
    const PROMPTS: Prompts = Prompts {
        skipped: &[
            "<system-reminder>",
            "<system-notification>",
            "<user-memory-input>",
            "<trodden-memory",
            "[Request interrupted",
            "Trodden: ",
        ],
        pasted: None,
    };

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "TodoWrite",
        "TodoRead",
        "ExitSpecMode",
        "AskUser",
        "AskUserQuestion",
        "GenerateDroid",
        "Skill",
    ];

    pub fn parse(text: &str, model: Option<&str>, redactor: &Redactor) -> Result<Trace> {
        let mut reader = Reader {
            builder: TraceBuilder::new(HARNESS, redactor),
            pending: Vec::new(),
        };
        if let Some(model) = model {
            reader.builder.model(model);
        }
        for value in text.lines().filter_map(Self::record) {
            reader.push(&value);
        }
        reader
            .builder
            .finish("no Factory Droid session found in transcript")
    }

    pub fn session(text: &str) -> Option<String> {
        text.lines()
            .filter_map(Self::record)
            .find_map(|record| Self::session_of(&record))
    }

    pub fn working_directory(text: &str) -> Option<String> {
        text.lines()
            .filter_map(Self::record)
            .filter(|record| Self::kind(record) == Some("session_start"))
            .find_map(|record| {
                Self::string(&record, "cwd")
                    .or_else(|| Self::string(&record, "lastCwd"))
                    .filter(|cwd| !cwd.is_empty())
                    .map(str::to_owned)
            })
    }

    fn record(line: &str) -> Option<Value> {
        serde_json::from_str::<Value>(line)
            .ok()
            .filter(Value::is_object)
    }

    fn kind(record: &Value) -> Option<&str> {
        record.get("type").and_then(Value::as_str)
    }

    fn string<'v>(value: &'v Value, key: &str) -> Option<&'v str> {
        value.get(key).and_then(Value::as_str)
    }

    fn session_of(record: &Value) -> Option<String> {
        let id = if Self::kind(record) == Some("session_start") {
            Self::string(record, "id")
        } else {
            Self::string(record, "session_id").or_else(|| Self::string(record, "sessionId"))
        };
        id.filter(|id| !id.is_empty()).map(str::to_owned)
    }

    fn timestamp(record: &Value) -> Option<Timestamp> {
        match record.get("timestamp")? {
            Value::String(text) => text.parse().ok(),
            Value::Number(number) => Timestamp::from_millisecond(number.as_i64()?).ok(),
            _ => None,
        }
    }
}

struct Reader<'a> {
    builder: TraceBuilder<'a>,
    pending: Vec<(String, Value)>,
}

impl Reader<'_> {
    fn push(&mut self, record: &Value) {
        if let Some(session) = Transcript::session_of(record) {
            self.builder.session(&session);
        }
        if let Some(at) = Transcript::timestamp(record) {
            self.builder.time_at(at);
        }
        match Transcript::kind(record) {
            Some("session_start") => {
                if let Some(cwd) = Transcript::string(record, "cwd")
                    .or_else(|| Transcript::string(record, "lastCwd"))
                {
                    self.builder.directory(cwd);
                }
            }
            Some("message") => self.push_message(record),
            Some("compaction_state") => self.builder.compaction(true),
            _ => {}
        }
    }

    fn push_message(&mut self, record: &Value) {
        let (role, content) = match record.get("message") {
            Some(message) => (
                Transcript::string(message, "role"),
                message.get("content").cloned().unwrap_or_default(),
            ),
            None => (
                Transcript::string(record, "role"),
                record.get("text").cloned().unwrap_or_default(),
            ),
        };
        if let Some(model) = record
            .get("message")
            .and_then(|message| Transcript::string(message, "model"))
        {
            self.builder.model(model);
        }
        let injected = record
            .get("visibility")
            .is_some_and(|value| !value.is_null());
        match role {
            Some("user") => self.push_user(&content, injected),
            Some("assistant") => self.push_assistant(&content),
            _ => {}
        }
    }

    fn push_user(&mut self, content: &Value, injected: bool) {
        let mut prompt = None;
        match content {
            Value::String(text) => prompt = Some(text.as_str()),
            Value::Array(blocks) => {
                for block in blocks {
                    match Transcript::string(block, "type") {
                        Some("tool_result") => self.resolve(block),
                        Some("text") if prompt.is_none() => {
                            prompt = Transcript::string(block, "text")
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
        if let Some(text) = prompt.filter(|_| !injected) {
            self.builder.prompt(text, &Transcript::PROMPTS);
        }
    }

    fn push_assistant(&mut self, content: &Value) {
        let Value::Array(blocks) = content else {
            return;
        };
        for block in blocks {
            if Transcript::string(block, "type") != Some("tool_use") {
                continue;
            }
            let (Some(id), Some(name)) = (
                Transcript::string(block, "id"),
                Transcript::string(block, "name"),
            ) else {
                continue;
            };
            let input = block.get("input").cloned().unwrap_or_default();
            if let Some((action, args)) = self.tool_call(name, &input) {
                self.builder.call(Some(id), name, action, args);
                self.pending.push((id.to_owned(), input));
            }
        }
    }

    fn field<'v>(input: &'v Value, keys: &[&str]) -> Option<&'v str> {
        keys.iter().find_map(|key| Transcript::string(input, key))
    }

    const PATH: &'static [&'static str] = &["file_path", "path", "filePath", "file"];

    fn tool_call(&self, name: &str, input: &Value) -> Option<(ToolAction, ToolArgs)> {
        if Transcript::BOOKKEEPING_TOOLS.contains(&name) {
            return None;
        }
        let builder = &self.builder;
        let mut args = ToolArgs::default();
        let action = match name {
            "Execute" => {
                let (shell, action) =
                    builder.shell(Self::field(input, &["command", "cmd"]).unwrap_or_default());
                args = shell;
                action
            }
            "Read" => {
                args.path = Self::field(input, Self::PATH).map(|p| builder.relative(p));
                ToolAction::Read
            }
            "LS" => {
                args.path = Self::field(input, &["directory_path", "path", "directory"])
                    .map(|p| builder.relative(p));
                ToolAction::Search
            }
            "Edit" | "MultiEdit" | "Create" | "Write" => {
                args.path = Self::field(input, Self::PATH).map(|p| builder.relative(p));
                ToolAction::Edit
            }
            "ApplyPatch" => {
                args.path = Patch::files(Self::patch(input))
                    .first()
                    .map(|(path, _, _)| builder.relative(path));
                ToolAction::Edit
            }
            "Grep" | "Glob" => {
                args.pattern = Self::field(input, &["pattern", "patterns", "query"])
                    .map(|p| builder.search_pattern(p, name == "Glob"));
                args.path = Self::field(input, &["path", "folder"]).map(|p| builder.relative(p));
                ToolAction::Search
            }
            "FetchUrl" | "WebFetch" => {
                args.url = Self::field(input, &["url"]).map(|u| builder.redact(u));
                ToolAction::Fetch
            }
            "WebSearch" => {
                args.query = Self::field(input, &["query"]).map(|q| builder.redact(q));
                ToolAction::Fetch
            }
            "Task" => ToolAction::Delegate,
            _ => ToolAction::Other,
        };
        Some((action, args))
    }

    fn patch(input: &Value) -> &str {
        match input {
            Value::String(text) => text,
            _ => Self::field(input, &["patch", "patchText", "input"]).unwrap_or_default(),
        }
    }

    fn resolve(&mut self, block: &Value) {
        let Some(id) = Transcript::string(block, "tool_use_id") else {
            return;
        };
        if !self.builder.is_pending(id) {
            return;
        }
        let input = self
            .pending
            .iter()
            .position(|(pending, _)| pending == id)
            .map(|index| self.pending.swap_remove(index).1)
            .unwrap_or_default();
        let text = Self::text(block.get("content"));
        let is_error = block.get("is_error").and_then(Value::as_bool) == Some(true);
        let exit_code = Self::exit_code(&text);
        let finish = if is_error && Self::declined(&text) {
            Finish::Interrupted
        } else if is_error || exit_code.is_some_and(|code| code != 0) {
            Finish::Failed { exit_code }
        } else {
            Finish::Succeeded
        };
        let changes = match (finish, self.builder.pending_call(id)) {
            (Finish::Succeeded, Some(call)) if call.action == ToolAction::Edit => {
                let tool = call.tool.clone();
                self.changes(&tool, &input)
            }
            _ => Vec::new(),
        };
        self.builder.resolve(id, finish, &text, None, changes);
    }

    fn text(content: Option<&Value>) -> String {
        match content {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| Transcript::string(part, "text"))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        }
    }

    fn declined(text: &str) -> bool {
        let lower = text.to_ascii_lowercase();
        [
            "rejected by the user",
            "user rejected",
            "cancelled by user",
            "canceled by user",
            "interrupted by user",
            "user denied",
        ]
        .iter()
        .any(|marker| lower.contains(marker))
    }

    fn exit_code(text: &str) -> Option<i32> {
        let lines: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        [lines.first(), lines.last()]
            .into_iter()
            .flatten()
            .find_map(|line| Self::exit_line(line))
    }

    fn exit_line(line: &str) -> Option<i32> {
        let lower = line.to_ascii_lowercase();
        let lower =
            lower.trim_matches(|c: char| matches!(c, '[' | ']' | '(' | ')' | '.' | '<' | '>'));
        for marker in [
            "exit code:",
            "exit code",
            "exited with code",
            "exit status",
            "return code:",
        ] {
            if let Some(at) = lower.find(marker) {
                let rest = lower[at + marker.len()..].trim_start_matches([' ', ':', '=']);
                let digits: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-')
                    .collect();
                if let Ok(code) = digits.parse() {
                    return Some(code);
                }
            }
        }
        None
    }

    fn changes(&self, tool: &str, input: &Value) -> Vec<FileChange> {
        let path = Self::field(input, Self::PATH);
        match tool {
            "Create" | "Write" => path
                .map(|path| {
                    let content = Self::field(input, &["content", "text"]).unwrap_or_default();
                    self.builder
                        .change(path, true, &Diff::created(content), None)
                })
                .into_iter()
                .collect(),
            "Edit" => path
                .map(|path| {
                    let old = Self::field(input, &["old_str", "old_string", "oldString"])
                        .unwrap_or_default();
                    let new = Self::field(input, &["new_str", "new_string", "newString"])
                        .unwrap_or_default();
                    self.builder
                        .change(path, false, &Diff::replaced(old, new), None)
                })
                .into_iter()
                .collect(),
            "MultiEdit" => path
                .map(|path| {
                    let mut diff = Diff::default();
                    for edit in input
                        .get("edits")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                    {
                        let old = Self::field(edit, &["old_str", "old_string", "oldString"])
                            .unwrap_or_default();
                        let new = Self::field(edit, &["new_str", "new_string", "newString"])
                            .unwrap_or_default();
                        let one = Diff::replaced(old, new);
                        diff.added = diff.added.saturating_add(one.added);
                        diff.removed = diff.removed.saturating_add(one.removed);
                        diff.changed.extend(one.changed);
                    }
                    self.builder.change(path, false, &diff, None)
                })
                .into_iter()
                .collect(),
            "ApplyPatch" => Patch::files(Self::patch(input))
                .into_iter()
                .map(|(path, created, body)| {
                    self.builder
                        .change(&path, created, &Diff::unified(&body), None)
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug)]
struct Patch;

impl Patch {
    fn files(text: &str) -> Vec<(String, bool, String)> {
        let mut files: Vec<(String, bool, String)> = Vec::new();
        for line in text.lines() {
            let header = [
                ("*** Add File:", true),
                ("*** Update File:", false),
                ("*** Delete File:", false),
            ]
            .into_iter()
            .find_map(|(prefix, created)| {
                line.strip_prefix(prefix).map(|path| (path.trim(), created))
            });
            if let Some((path, created)) = header {
                files.push((path.to_owned(), created, String::new()));
            } else if line.starts_with("*** ") {
                continue;
            } else if let Some((_, _, body)) = files.last_mut() {
                body.push_str(line);
                body.push('\n');
            }
        }
        files
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use trodden_core::trace::{EventKind, ToolCall, ToolOutcome};

    use super::*;

    const CWD: &str = "/home/dev/shop";

    #[derive(Debug, Default)]
    struct Session {
        lines: Vec<String>,
    }

    impl Session {
        fn new() -> Self {
            let mut session = Self::default();
            session.record(
                json!({"type": "session_start", "id": "6f0c2b1e-0d1c-4c1e-9d55-2e7d1f3a9c20",
                                  "title": "Fix paging bug", "cwd": CWD, "version": 2}),
            );
            session
        }

        fn record(&mut self, value: Value) -> &mut Self {
            self.lines.push(value.to_string());
            self
        }

        fn message(&mut self, at: Value, role: &str, content: Value) -> &mut Self {
            self.record(
                json!({"type": "message", "id": format!("m{}", self.lines.len()), "timestamp": at,
                               "message": {"role": role, "content": content}}),
            )
        }

        fn prompt(&mut self, text: &str) -> &mut Self {
            self.message(
                json!("2026-10-09T10:00:00.000Z"),
                "user",
                json!([{"type": "text", "text": text}]),
            )
        }

        fn tool(
            &mut self,
            id: &str,
            name: &str,
            input: Value,
            result: Value,
            is_error: bool,
        ) -> &mut Self {
            self.message(
                json!("2026-10-09T10:00:03.000Z"),
                "assistant",
                json!([{"type": "tool_use", "id": id, "name": name, "input": input}]),
            );
            self.message(
                json!(1_791_540_000_000_i64),
                "user",
                json!([{"type": "tool_result", "tool_use_id": id, "content": result, "is_error": is_error}]),
            )
        }

        fn text(&self) -> String {
            self.lines.iter().map(|line| format!("{line}\n")).collect()
        }

        fn trace(&self) -> Trace {
            Transcript::parse(
                &self.text(),
                Some("claude-opus-4-5"),
                &Redactor::with_home("/home/dev"),
            )
            .expect("valid transcript")
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
    }

    #[test]
    fn a_session_reads_as_a_trace() {
        let mut session = Session::new();
        session
            .prompt("Fix the paging bug in src/paginate.js\nPage 2 repeats an item")
            .tool("t1", "Read", json!({"file_path": format!("{CWD}/src/paginate.js")}), json!("..."), false)
            .tool(
                "t2",
                "Edit",
                json!({"file_path": format!("{CWD}/src/paginate.js"),
                       "old_str": "function paginate(items) {\n  return items.slice(start, end + 1);\n}",
                       "new_str": "function paginate(items) {\n  return items.slice(start, end);\n}"}),
                json!("File edited"),
                false,
            )
            .tool("t3", "Execute", json!({"command": "npm test"}), json!("1 passing"), false)
            .record(json!({"type": "compaction_state", "id": "c1", "timestamp": 1_791_540_100_000_i64, "summaryText": "..."}));

        let trace = session.trace();
        let calls = session.calls();

        assert_eq!(
            trace.session.as_str(),
            "6f0c2b1e-0d1c-4c1e-9d55-2e7d1f3a9c20"
        );
        assert_eq!(trace.harness.as_str(), "droid");
        assert_eq!(trace.cwd, "~/shop");
        assert_eq!(trace.model.as_deref(), Some("claude-opus-4-5"));
        assert_eq!(session.prompts(), ["Fix the paging bug in src/paginate.js"]);
        assert_eq!(calls[0].action, ToolAction::Read);
        assert_eq!(calls[0].args.path.as_deref(), Some("src/paginate.js"));
        assert_eq!(calls[1].changes[0].path, "src/paginate.js");
        assert_eq!(
            (
                calls[1].changes[0].lines_added,
                calls[1].changes[0].lines_removed
            ),
            (1, 1)
        );
        assert_eq!(calls[2].args.command.as_deref(), Some("npm test"));
        assert_eq!(calls[2].outcome, ToolOutcome::Succeeded);
        assert!(matches!(
            trace.events.last().map(|event| &event.kind),
            Some(EventKind::Compaction { .. })
        ));
        assert_eq!(
            Transcript::working_directory(&session.text()).as_deref(),
            Some(CWD)
        );
    }

    #[test]
    fn failed_commands_keep_their_exit_code_and_error() {
        let mut session = Session::new();
        session
            .prompt("Fix the build")
            .tool("t1", "Execute", json!({"command": "npm test"}),
                  json!("Error: Cannot find module 'left-pad'\nExit code: 1"), true)
            .tool("t2", "Execute", json!({"command": "cargo build"}),
                  json!([{"type": "text", "text": "error[E0425]: cannot find value `x`\n[Process exited with code 101]"}]), false)
            .tool("t3", "Execute", json!({"command": "npm run lint"}), json!("lint failed"), true)
            .tool("t4", "Execute", json!({"command": "rm -rf build"}), json!("Command rejected by the user"), true);

        let calls = session.calls();

        assert_eq!(calls[0].outcome, ToolOutcome::Failed { exit_code: Some(1) });
        assert!(calls[0].error.is_some());
        assert_eq!(
            calls[1].outcome,
            ToolOutcome::Failed {
                exit_code: Some(101)
            }
        );
        assert_eq!(calls[2].outcome, ToolOutcome::Failed { exit_code: None });
        assert_eq!(calls[3].outcome, ToolOutcome::Interrupted);
    }

    #[test]
    fn exit_codes_mid_output_are_not_the_command_s() {
        assert_eq!(
            Reader::exit_code("running\ntest exit code 1 handled ... ok\ndone"),
            None
        );
        assert_eq!(Reader::exit_code("Exit code 2\nboom"), Some(2));
        assert_eq!(Reader::exit_code("boom\nexit status 3"), Some(3));
    }

    #[test]
    fn injected_rows_are_not_prompts() {
        let mut session = Session::new();
        session
            .record(json!({"type": "message", "id": "h1", "visibility": "hidden", "timestamp": "2026-10-09T10:00:00Z",
                           "message": {"role": "user", "content": [{"type": "text", "text": "SessionStart hook output"}]}}))
            .prompt("<system-reminder>\nThe user opened README.md\n</system-reminder>")
            .prompt("Rename the shipping helper")
            .prompt("<trodden-memory id=\"p\" rev=\"1\">\nsteps\n</trodden-memory>")
            .prompt("Trodden: the procedure recalled for this task is checked with `npm test`")
            .record(json!({"type": "message", "id": "f1", "role": "user", "text": "Also update the changelog",
                           "session_id": "6f0c2b1e-0d1c-4c1e-9d55-2e7d1f3a9c20"}));

        assert_eq!(
            session.prompts(),
            ["Rename the shipping helper", "Also update the changelog"]
        );
    }

    #[test]
    fn interrupted_calls_stay_interrupted() {
        let mut session = Session::new();
        session.prompt("Run the tests").message(
            json!("2026-10-09T10:00:03Z"),
            "assistant",
            json!([{"type": "tool_use", "id": "t1", "name": "Execute", "input": {"command": "npm test"}}]),
        );

        assert_eq!(session.calls()[0].outcome, ToolOutcome::Interrupted);
    }

    #[test]
    fn created_files_and_patches_count_their_lines() {
        let mut session = Session::new();
        session
            .prompt("Add a changelog")
            .tool("t1", "Create", json!({"file_path": format!("{CWD}/CHANGELOG.md"), "content": "# Changes\n\n- paging\n"}),
                  json!("Created"), false)
            .tool("t2", "ApplyPatch",
                  json!({"patch": "*** Begin Patch\n*** Update File: src/cart.js\n@@ function total\n-  return sum;\n+  return sum + tax;\n*** Add File: src/tax.js\n+export const RATE = 0.2;\n*** End Patch\n"}),
                  json!("Applied"), false)
            .tool("t3", "Edit", json!({"file_path": format!("{CWD}/src/a.js"), "old_str": "a", "new_str": "b"}),
                  json!("Error: old_str not found"), true);

        let calls = session.calls();

        assert!(calls[0].changes[0].created);
        assert_eq!(calls[0].changes[0].lines_added, 3);
        assert_eq!(calls[1].args.path.as_deref(), Some("src/cart.js"));
        assert_eq!(calls[1].changes.len(), 2);
        assert_eq!(
            (
                calls[1].changes[0].lines_added,
                calls[1].changes[0].lines_removed
            ),
            (1, 1)
        );
        assert!(calls[1].changes[1].created);
        assert!(calls[2].changes.is_empty());
    }

    #[test]
    fn malformed_lines_and_unknown_rows_are_skipped() {
        let mut session = Session::new();
        session.prompt("Fix the build");
        session.lines.push("{\"type\":\"message\",\"tru".to_owned());
        session.lines.push("[1, 2]".to_owned());
        session
            .record(json!({"type": "todo_state", "id": "x", "todos": {"todos": []}}))
            .record(json!({"type": "message", "id": "p", "message": {"verdict": "allow"}}))
            .record(json!({"type": "message", "id": "q", "message": {"role": "user", "content": [{"type": "tool_result"}]}}))
            .record(json!({"type": "session_end", "durationMs": 1200}));

        assert_eq!(session.prompts(), ["Fix the build"]);
        assert!(Transcript::parse("not json\n", None, &Redactor::with_home("/home/dev")).is_err());
    }

    #[test]
    fn subagent_messages_without_a_header_still_name_their_session() {
        let line = json!({"type": "message", "id": "f1", "role": "user", "text": "Look up the tax rate",
                          "session_id": "sub-1"});

        assert_eq!(
            Transcript::session(&format!("{line}\n")).as_deref(),
            Some("sub-1")
        );
    }
}
