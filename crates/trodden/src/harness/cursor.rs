use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use jiff::Timestamp;
use serde::Deserialize;
use serde_json::{Value, json};
use trodden_capture::{
    Reminder,
    journal::{Edit, Journal, Observation, Observer, Ran},
};
use trodden_core::{Trace, trace::ToolAction};
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply};
use crate::connect::{HookFile, Program};

#[derive(Debug, Deserialize)]
struct CursorHookInput {
    hook_event_name: String,
    #[serde(default)]
    conversation_id: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    workspace_roots: Vec<PathBuf>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    model_id: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_input: Option<Value>,
    #[serde(default)]
    tool_output: Option<Value>,
    #[serde(default)]
    duration: Option<Value>,
    #[serde(default)]
    error_message: Option<String>,
    #[serde(default)]
    failure_type: Option<String>,
    #[serde(default)]
    is_interrupt: bool,
    #[serde(default)]
    parent_tool_call_id: Option<String>,
    #[serde(default)]
    file_path: Option<String>,
    #[serde(default)]
    edits: Vec<CursorEdit>,
    #[serde(default)]
    loop_count: Option<u64>,
    #[serde(default)]
    trigger: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CursorEdit {
    #[serde(default)]
    old_string: String,
    #[serde(default)]
    new_string: String,
}

#[derive(Debug, Default, PartialEq, Eq)]
struct ShellResult {
    exit_code: Option<i32>,
    output: String,
}

#[derive(Debug)]
pub(crate) struct Cursor;

impl Cursor {
    const HARNESS: &str = "cursor";

    const SHELL: &str = "Shell";

    const TIMEOUT_SECONDS: u64 = 5;

    const EDIT_TOOLS: &[&str] = &[
        "Write",
        "StrReplace",
        "Edit",
        "MultiEdit",
        "edit_file",
        "search_replace",
        "ApplyPatch",
    ];

    const BOOKKEEPING_TOOLS: &[&str] = &["TodoWrite", "todo_write", "AskQuestion", "SwitchMode"];

    const POST_TOOL_MATCHER: &str = "Shell|Read|Grep|Glob|Delete|Task|WebFetch|WebSearch";

    const EVENTS: &[(&str, Option<&str>)] = &[
        ("sessionStart", None),
        ("beforeSubmitPrompt", None),
        ("postToolUse", Some(Self::POST_TOOL_MATCHER)),
        ("postToolUseFailure", Some(Self::SHELL)),
        ("afterFileEdit", None),
        ("stop", None),
        ("preCompact", None),
        ("sessionEnd", None),
    ];

    fn config() -> Result<PathBuf> {
        Ok(env::home_dir()
            .context("find the home directory")?
            .join(".cursor"))
    }

    fn hooks(config: &Path) -> PathBuf {
        config.join("hooks.json")
    }

    fn connect_at(config: &Path, program: &Program) -> Result<Vec<Change>> {
        let mut file = HookFile::load(Self::hooks(config))?;
        file.remove(&["hooks"], Self::HARNESS);
        if file.get("version").is_none() {
            file.set("version", json!(1))?;
        }
        let command = program.hook(Self::HARNESS);
        for (event, matcher) in Self::EVENTS {
            let mut handler = json!({
                "type": "command",
                "command": command,
                "timeout": Self::TIMEOUT_SECONDS,
            });
            if let Some(matcher) = matcher {
                handler["matcher"] = json!(matcher);
            }
            if *event == "stop" {
                handler["loop_limit"] = json!(1);
            }
            file.add(&["hooks"], event, handler)?;
        }
        Ok(file.change()?.into_iter().collect())
    }

    fn disconnect_at(config: &Path) -> Result<Vec<Change>> {
        let mut file = HookFile::load(Self::hooks(config))?;
        file.remove(&["hooks"], Self::HARNESS);
        Ok(file.change()?.into_iter().collect())
    }

    fn connected_at(config: &Path) -> Result<bool> {
        Ok(HookFile::load(Self::hooks(config))?.contains(&["hooks"], Self::HARNESS))
    }

    fn string<'v>(value: Option<&'v Value>, keys: &[&str]) -> Option<&'v str> {
        let value = value?;
        keys.iter()
            .find_map(|key| value.get(key).and_then(Value::as_str))
            .filter(|text| !text.is_empty())
    }

    fn shell_result(output: Option<&Value>) -> ShellResult {
        let decoded = match output {
            Some(Value::String(text)) => match serde_json::from_str::<Value>(text) {
                Ok(value @ Value::Object(_)) => value,
                _ => {
                    return ShellResult {
                        exit_code: None,
                        output: text.clone(),
                    };
                }
            },
            Some(value @ Value::Object(_)) => value.clone(),
            _ => return ShellResult::default(),
        };
        let exit_code = ["exitCode", "exit_code", "code"]
            .iter()
            .find_map(|key| decoded.get(key).and_then(Value::as_i64))
            .and_then(|code| i32::try_from(code).ok());
        let output: Vec<&str> = ["stdout", "stderr", "output", "error"]
            .iter()
            .filter_map(|key| decoded.get(key).and_then(Value::as_str))
            .filter(|text| !text.trim().is_empty())
            .collect();
        ShellResult {
            exit_code,
            output: output.join("\n"),
        }
    }

    fn patch(edits: &[CursorEdit]) -> String {
        let mut patch = String::new();
        for edit in edits {
            let old: Vec<&str> = edit.old_string.lines().collect();
            let new: Vec<&str> = edit.new_string.lines().collect();
            let prefix = old.iter().zip(&new).take_while(|(a, b)| a == b).count();
            let suffix = old[prefix..]
                .iter()
                .rev()
                .zip(new[prefix..].iter().rev())
                .take_while(|(a, b)| a == b)
                .count();
            patch.push_str("@@\n");
            for line in &old[prefix..old.len() - suffix] {
                patch.push_str(&format!("-{line}\n"));
            }
            for line in &new[prefix..new.len() - suffix] {
                patch.push_str(&format!("+{line}\n"));
            }
        }
        patch
    }

    fn observe_tool(
        input: &CursorHookInput,
        observer: &Observer<'_>,
        failed: bool,
    ) -> Option<Observation> {
        let tool = input.tool_name.as_deref()?;
        if Self::EDIT_TOOLS.contains(&tool) || Self::BOOKKEEPING_TOOLS.contains(&tool) {
            return None;
        }
        let args = input.tool_input.as_ref();
        let path = Self::string(args, &["path", "file_path", "target_file", "filePath"]);
        Some(match tool {
            Self::SHELL => return None,
            "Read" | "read_file" => observer.read(tool, path?),
            "Grep" | "Glob" | "grep_search" | "file_search" | "codebase_search" | "list_dir" => {
                observer.search(
                    tool,
                    Self::string(args, &["pattern", "query", "glob_pattern", "glob"]),
                    Self::string(args, &["path", "target_directory", "directory"]),
                )
            }
            "Delete" | "delete_file" => observer.edit(tool, path?, Edit::Unknown, !failed),
            "WebFetch" | "WebSearch" | "web_search" => observer.fetch(
                tool,
                Self::string(args, &["url"]),
                Self::string(args, &["query", "search_term"]),
            ),
            "Task" => observer.other(tool, ToolAction::Delegate),
            _ => observer.other(tool, ToolAction::Other),
        })
    }
}

impl Agent for Cursor {
    fn title(&self) -> &'static str {
        "Cursor"
    }

    fn journaled(&self) -> bool {
        true
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(None)
    }

    fn transcripts(&self, _history: &Path) -> Result<Vec<PathBuf>> {
        Ok(Vec::new())
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        Ok((
            Journal::parse(text, Self::HARNESS, redactor)?,
            Journal::working_directory(text).map(PathBuf::from),
        ))
    }

    fn event(&self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>> {
        let input: CursorHookInput =
            serde_json::from_str(payload).context("parse the hook payload")?;
        let Some(session) = input
            .conversation_id
            .clone()
            .or_else(|| input.session_id.clone())
            .filter(|session| !session.is_empty())
        else {
            return Ok(None);
        };
        let Some(cwd) = input
            .cwd
            .clone()
            .filter(|cwd| cwd.is_absolute())
            .or_else(|| input.workspace_roots.first().cloned())
            .filter(|cwd| cwd.is_absolute())
        else {
            return Ok(None);
        };
        let root = cwd.to_string_lossy().into_owned();
        let here = Self::string(input.tool_input.as_ref(), &["working_directory", "cwd"])
            .filter(|dir| Path::new(dir).is_absolute())
            .map_or_else(|| root.clone(), str::to_owned);
        let model = input.model_id.as_deref().or(input.model.as_deref());
        let observer = Observer {
            session: &session,
            cwd: &root,
            at: Timestamp::now(),
            model,
            redactor,
        };
        let shell = Observer {
            cwd: &here,
            ..observer
        };
        let duration_ms = input.duration.as_ref().and_then(Value::as_u64);
        let is_shell = input.tool_name.as_deref() == Some(Self::SHELL);
        let command = Self::string(input.tool_input.as_ref(), &["command"]).unwrap_or_default();
        let subagent = input.parent_tool_call_id.is_some();

        let mut observed = Vec::new();
        let moment = match input.hook_event_name.as_str() {
            "sessionStart" => Moment::SessionStart,
            "beforeSubmitPrompt" => {
                let Some(prompt) = input
                    .prompt
                    .as_deref()
                    .filter(|prompt| !prompt.trim().is_empty())
                else {
                    return Ok(None);
                };
                if prompt.trim_start().starts_with(Reminder::PREFIX) {
                    return Ok(None);
                }
                observed.push(observer.prompt(prompt));
                Moment::Prompt(prompt.to_owned())
            }
            "postToolUse" if is_shell => {
                let result = Self::shell_result(input.tool_output.as_ref());
                let failed = result.exit_code.is_some_and(|code| code != 0);
                if !subagent {
                    observed.push(shell.command(
                        Self::SHELL,
                        command,
                        Ran {
                            output: &result.output,
                            exit_code: result.exit_code,
                            failed,
                            interrupted: false,
                            duration_ms,
                        },
                    ));
                }
                if failed {
                    Moment::CommandFailed(result.output)
                } else {
                    Moment::ToolDone
                }
            }
            "postToolUse" => {
                observed.extend(Self::observe_tool(&input, &observer, false).filter(|_| !subagent));
                Moment::ToolDone
            }
            "postToolUseFailure" if is_shell => {
                let error = input.error_message.clone().unwrap_or_default();
                let interrupted = input.is_interrupt
                    || input.failure_type.as_deref() == Some("permission_denied");
                if !subagent {
                    observed.push(shell.command(
                        Self::SHELL,
                        command,
                        Ran {
                            output: &error,
                            exit_code: None,
                            failed: true,
                            interrupted,
                            duration_ms,
                        },
                    ));
                }
                if interrupted {
                    Moment::ToolDone
                } else {
                    Moment::CommandFailed(error)
                }
            }
            "afterFileEdit" => {
                let Some(path) = input.file_path.as_deref().filter(|path| !path.is_empty()) else {
                    return Ok(None);
                };
                let patch = Cursor::patch(&input.edits);
                let edit = match input.edits.as_slice() {
                    [only] if only.old_string.is_empty() => Edit::Created {
                        content: &only.new_string,
                    },
                    [] => Edit::Unknown,
                    _ => Edit::Patched { diff: &patch },
                };
                observed.push(observer.edit("Edit", path, edit, true));
                Moment::ToolDone
            }
            "stop" => Moment::TurnEnd {
                continued: input.loop_count.is_some_and(|count| count > 0),
            },
            "preCompact" => {
                observed.push(observer.compaction(input.trigger.as_deref() != Some("manual")));
                Moment::Compacting
            }
            "sessionEnd" => Moment::SessionEnd,
            _ => return Ok(None),
        };
        Ok(Some(HookEvent {
            name: input.hook_event_name,
            session,
            cwd,
            transcript: None,
            moment,
            observed,
        }))
    }

    // `beforeSubmitPrompt` cannot add context, so prompt recall is never shown, and never
    // recorded.
    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::CommandFailed(_), Reply::Recall(envelope)) => {
                Some(json!({ "additional_context": envelope }).to_string())
            }
            (Moment::TurnEnd { continued: false }, Reply::Remind(reason)) => {
                Some(json!({ "followup_message": reason }).to_string())
            }
            _ => None,
        }
    }

    fn detected(&self) -> bool {
        Self::config().is_ok_and(|config| config.is_dir())
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        Self::connect_at(&Self::config()?, program)
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        Self::disconnect_at(&Self::config()?)
    }

    fn connected(&self) -> Result<bool> {
        Self::connected_at(&Self::config()?)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, process};

    use trodden_capture::journal::Observed;
    use trodden_core::trace::{EventKind, ToolOutcome};

    use super::*;

    const ROOT: &str = "/home/dev/shop";

    #[derive(Debug)]
    struct Payload;

    impl Payload {
        fn event(extra: Value) -> Option<HookEvent> {
            let mut payload = json!({
                "conversation_id": "6130c625-1b9e-4f7a-a1a2-0c9d2c1f4e11",
                "generation_id": "b0f7d2e4",
                "model": "claude-opus-4-7-thinking-max",
                "cursor_version": "3.23.1",
                "workspace_roots": [ROOT],
                "user_email": null,
                "transcript_path": null,
            });
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            Cursor
                .event(&payload.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
        }

        fn moment(extra: Value) -> Option<Moment> {
            Self::event(extra).map(|event| event.moment)
        }

        fn shell(command: &str, output: &str, cwd: &str) -> Value {
            json!({"hook_event_name": "postToolUse", "tool_name": "Shell",
                   "tool_input": {"command": command, "working_directory": cwd},
                   "tool_output": output, "tool_use_id": "toolu_1", "cwd": cwd, "duration": 5432})
        }

        fn rendered(moment: Moment, reply: Reply<'_>) -> Option<String> {
            let event = HookEvent {
                name: String::new(),
                session: "s".to_owned(),
                cwd: PathBuf::from(ROOT),
                transcript: None,
                moment,
                observed: Vec::new(),
            };
            Cursor.render(&event, reply)
        }
    }

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-cursor-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn hooks(&self) -> Value {
            serde_json::from_str(
                &fs::read_to_string(self.dir.join("hooks.json")).expect("readable"),
            )
            .expect("JSON")
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn prompts_are_journaled_but_never_shown() {
        let event = Payload::event(json!({"hook_event_name": "beforeSubmitPrompt",
                                          "prompt": "Fix the paging bug", "attachments": []}))
        .expect("event");

        assert_eq!(
            event.moment,
            Moment::Prompt("Fix the paging bug".to_owned())
        );
        assert_eq!(event.session, "6130c625-1b9e-4f7a-a1a2-0c9d2c1f4e11");
        assert_eq!(event.cwd, PathBuf::from(ROOT));
        assert!(matches!(
            &event.observed[..],
            [Observation {
                observed: Observed::Prompt { .. },
                ..
            }]
        ));
        assert_eq!(
            event.observed[0].model.as_deref(),
            Some("claude-opus-4-7-thinking-max")
        );
        assert_eq!(Payload::rendered(event.moment, Reply::Recall("<m/>")), None);
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "beforeSubmitPrompt",
                                   "prompt": "Trodden: the procedure recalled for this task is checked with `npm test`"})),
            None
        );
    }

    #[test]
    fn shell_failures_are_recalled_from_either_event() {
        let failed = Payload::event(Payload::shell(
            "npm test",
            r#"{"exitCode":1,"stdout":"","stderr":"Error: Cannot find module 'left-pad'"}"#,
            ROOT,
        ))
        .expect("event");
        let passed = Payload::moment(Payload::shell(
            "npm test",
            r#"{"exitCode":0,"stdout":"1 passing"}"#,
            ROOT,
        ));
        let failure = |extra: Value| {
            let mut payload = json!({"hook_event_name": "postToolUseFailure", "tool_name": "Shell",
                                     "tool_input": {"command": "npm test"}, "cwd": ROOT,
                                     "error_message": "Command failed: boom", "failure_type": "error"});
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            Payload::moment(payload)
        };

        assert_eq!(
            failed.moment,
            Moment::CommandFailed("Error: Cannot find module 'left-pad'".to_owned())
        );
        let Observed::Call {
            outcome,
            duration_ms,
            ..
        } = &failed.observed[0].observed
        else {
            panic!("a call is journaled");
        };
        assert_eq!(*outcome, ToolOutcome::Failed { exit_code: Some(1) });
        assert_eq!(*duration_ms, Some(5432));
        assert_eq!(passed, Some(Moment::ToolDone));
        assert_eq!(
            failure(json!({})),
            Some(Moment::CommandFailed("Command failed: boom".to_owned()))
        );
        assert_eq!(
            failure(json!({"is_interrupt": true})),
            Some(Moment::ToolDone)
        );
        assert_eq!(
            failure(json!({"failure_type": "permission_denied"})),
            Some(Moment::ToolDone)
        );

        let rendered =
            Payload::rendered(Moment::CommandFailed("e".to_owned()), Reply::Recall("<m/>"))
                .expect("failure recall renders");
        assert_eq!(
            serde_json::from_str::<Value>(&rendered).expect("JSON"),
            json!({"additional_context": "<m/>"})
        );
    }

    #[test]
    fn odd_tool_outputs_are_tolerated() {
        assert_eq!(
            Cursor::shell_result(Some(&json!("plain text"))),
            ShellResult {
                exit_code: None,
                output: "plain text".to_owned()
            }
        );
        assert_eq!(
            Cursor::shell_result(Some(&json!({"exit_code": 2, "output": "boom"}))),
            ShellResult {
                exit_code: Some(2),
                output: "boom".to_owned()
            }
        );
        assert_eq!(Cursor::shell_result(None), ShellResult::default());
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "postToolUse", "tool_name": "Shell"})),
            Some(Moment::ToolDone)
        );
    }

    #[test]
    fn a_session_journals_into_a_trace() {
        let events = [
            json!({"hook_event_name": "beforeSubmitPrompt", "prompt": "Page 2 repeats the last product. Fix it."}),
            json!({"hook_event_name": "postToolUse", "tool_name": "Read", "cwd": ROOT,
                   "tool_input": {"path": format!("{ROOT}/src/paginate.js")}, "tool_output": "{}"}),
            json!({"hook_event_name": "afterFileEdit", "file_path": format!("{ROOT}/src/paginate.js"),
                   "edits": [{"old_string": "export function paginate(items) {\n  return items.slice(start, end + 1);\n}",
                              "new_string": "export function paginate(items) {\n  return items.slice(start, end);\n}"},
                             {"old_string": "// todo", "new_string": "// paging\n// done"}]}),
            json!({"hook_event_name": "postToolUse", "tool_name": "Write", "cwd": ROOT,
                   "tool_input": {"path": format!("{ROOT}/src/paginate.js")}}),
            Payload::shell(
                "npm test",
                r#"{"exitCode":0,"stdout":"1 passing"}"#,
                &format!("{ROOT}/web"),
            ),
            json!({"hook_event_name": "postToolUse", "tool_name": "Shell", "parent_tool_call_id": "toolu_0",
                   "tool_input": {"command": "ls"}, "tool_output": "{\"exitCode\":0}", "cwd": ROOT}),
            json!({"hook_event_name": "preCompact", "trigger": "auto", "context_usage_percent": 91}),
            json!({"hook_event_name": "stop", "status": "completed", "loop_count": 0}),
        ];
        let mut journal = String::new();
        for event in events {
            for observation in Payload::event(event).expect("event").observed {
                journal.push_str(&observation.to_line().expect("line encodes"));
            }
        }

        let (trace, cwd) = Cursor
            .parse(&journal, &Redactor::with_home("/home/dev"))
            .expect("journal parses");
        let calls: Vec<_> = trace
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                EventKind::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect();

        assert_eq!(cwd, Some(PathBuf::from(ROOT)));
        assert_eq!(trace.harness.as_str(), "cursor");
        assert_eq!(calls.len(), 3, "{calls:?}");
        assert_eq!(calls[0].action, ToolAction::Read);
        assert_eq!(calls[0].args.path.as_deref(), Some("src/paginate.js"));
        assert_eq!(
            (
                calls[1].changes[0].lines_added,
                calls[1].changes[0].lines_removed
            ),
            (3, 2)
        );
        assert_eq!(calls[2].args.command.as_deref(), Some("cd web && npm test"));
        assert!(matches!(
            trace.events.last().map(|event| &event.kind),
            Some(EventKind::Compaction { automatic: true })
        ));
    }

    #[test]
    fn stops_continue_only_once() {
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "stop", "status": "completed", "loop_count": 0})
            ),
            Some(Moment::TurnEnd { continued: false })
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "stop", "status": "completed", "loop_count": 1})
            ),
            Some(Moment::TurnEnd { continued: true })
        );
        assert_eq!(
            Payload::rendered(
                Moment::TurnEnd { continued: false },
                Reply::Remind("run it")
            ),
            Some(r#"{"followup_message":"run it"}"#.to_owned())
        );
        assert_eq!(
            Payload::rendered(Moment::TurnEnd { continued: true }, Reply::Remind("run it")),
            None
        );
    }

    #[test]
    fn other_events_and_odd_payloads_are_ignored() {
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "sessionEnd", "session_id": "x", "reason": "completed"})
            ),
            Some(Moment::SessionEnd)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "sessionStart", "session_id": "x"})),
            Some(Moment::SessionStart)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "afterAgentResponse", "text": "done"})),
            None
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "afterShellExecution", "command": "ls"})),
            None
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "stop", "workspace_roots": []})),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "stop", "conversation_id": "", "workspace_roots": [ROOT]})
            ),
            None
        );
        assert!(
            Cursor
                .event("not json", &Redactor::with_home("/home/dev"))
                .is_err()
        );
    }

    #[test]
    fn connect_keeps_other_hooks_and_round_trips() {
        let scratch = Scratch::new("connect");
        fs::write(
            scratch.dir.join("hooks.json"),
            json!({"version": 1, "hooks": {"stop": [{"command": "afplay done.aiff"}]}}).to_string(),
        )
        .expect("writable");
        let program = Program::at("/opt/trodden/bin/trodden");

        Scratch::apply(Cursor::connect_at(&scratch.dir, &program).expect("connects"));
        assert!(
            Cursor::connect_at(&scratch.dir, &program)
                .expect("connects")
                .is_empty(),
            "idempotent"
        );
        assert!(Cursor::connected_at(&scratch.dir).expect("readable"));
        let hooks = scratch.hooks();
        assert_eq!(hooks["version"], 1);
        assert_eq!(
            hooks["hooks"]["stop"][0],
            json!({"command": "afplay done.aiff"})
        );
        assert_eq!(
            hooks["hooks"]["stop"][1],
            json!({"type": "command", "command": "/opt/trodden/bin/trodden hook cursor", "timeout": 5, "loop_limit": 1})
        );
        assert_eq!(hooks["hooks"]["postToolUseFailure"][0]["matcher"], "Shell");
        assert!(hooks["hooks"].get("afterShellExecution").is_none());

        Scratch::apply(Cursor::disconnect_at(&scratch.dir).expect("disconnects"));
        assert_eq!(
            scratch.hooks(),
            json!({"version": 1, "hooks": {"stop": [{"command": "afplay done.aiff"}]}})
        );
        assert!(!Cursor::connected_at(&scratch.dir).expect("readable"));

        let fresh = Scratch::new("fresh");
        Scratch::apply(Cursor::connect_at(&fresh.dir, &program).expect("connects"));
        assert_eq!(fresh.hooks()["version"], 1);
    }
}
