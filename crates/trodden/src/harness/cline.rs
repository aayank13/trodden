use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use jiff::Timestamp;
use serde::Deserialize;
use serde_json::{Value, json};
use trodden_capture::journal::{Edit, Journal, Observation, Observer, Ran};
use trodden_core::{Trace, trace::ToolAction};
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply};
use crate::connect::{HookFile, Program};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClineHookInput {
    hook_name: String,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    session_context: Option<SessionContext>,
    #[serde(default)]
    workspace_roots: Vec<PathBuf>,
    #[serde(default)]
    workspace_info: Option<Value>,
    #[serde(default)]
    timestamp: Option<Value>,
    #[serde(default)]
    model: Option<Value>,
    #[serde(default, rename = "parent_agent_id")]
    parent_agent_id: Option<String>,
    #[serde(default)]
    user_prompt_submit: Option<PromptSubmit>,
    #[serde(default)]
    post_tool_use: Option<PostToolUse>,
    #[serde(default, rename = "tool_result")]
    tool_result: Option<ToolResult>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SessionContext {
    #[serde(default)]
    root_session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PromptSubmit {
    #[serde(default)]
    prompt: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct PostToolUse {
    tool_name: String,
    #[serde(default)]
    parameters: HashMap<String, Value>,
    #[serde(default)]
    result: Option<Value>,
    #[serde(default)]
    success: Option<bool>,
    #[serde(default)]
    execution_time_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ToolResult {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<Value>,
    #[serde(default)]
    output: Option<Value>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    duration_ms: Option<u64>,
}

#[derive(Debug)]
struct ToolUse {
    name: String,
    input: HashMap<String, Value>,
    output: String,
    succeeded: bool,
    duration_ms: Option<u64>,
}

impl ToolUse {
    const DENIED: &str = "The user denied this operation";

    fn of(input: &ClineHookInput) -> Option<Self> {
        let post = input.post_tool_use.as_ref();
        let raw = input.tool_result.as_ref();
        let name = post
            .map(|post| post.tool_name.clone())
            .or_else(|| raw.and_then(|raw| raw.name.clone()))?;
        let mut arguments: HashMap<String, Value> =
            post.map(|post| post.parameters.clone()).unwrap_or_default();
        if let Some(Value::Object(fields)) = raw.and_then(|raw| raw.input.as_ref()) {
            arguments.extend(fields.clone());
        }
        let error = raw
            .and_then(|raw| raw.error.as_ref())
            .map(Self::text)
            .filter(|error| !error.is_empty());
        let output = post
            .and_then(|post| post.result.as_ref())
            .or_else(|| raw.and_then(|raw| raw.output.as_ref()))
            .map(Self::text)
            .unwrap_or_default();
        let output = match &error {
            Some(error) if !output.contains(error.as_str()) => {
                format!("{output}\n{error}").trim().to_owned()
            }
            _ => output,
        };
        let succeeded = post
            .and_then(|post| post.success)
            .unwrap_or(error.is_none());
        Some(Self {
            name,
            input: arguments,
            output,
            succeeded,
            duration_ms: post
                .and_then(|post| post.execution_time_ms)
                .or_else(|| raw.and_then(|raw| raw.duration_ms)),
        })
    }

    fn text(value: &Value) -> String {
        match value {
            Value::String(text) => text.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }

    fn param(&self, key: &str) -> Option<String> {
        match self.input.get(key)? {
            Value::String(text) => Some(text.clone()),
            Value::Null => None,
            other => Some(other.to_string()),
        }
    }

    fn command(&self) -> Option<String> {
        if let Some(command) = self.param("command") {
            return Some(command);
        }
        let commands = match self.input.get("commands")? {
            Value::String(text) => {
                serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.clone()))
            }
            other => other.clone(),
        };
        match commands {
            Value::String(text) => Some(text),
            Value::Array(items) => {
                let commands: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
                (!commands.is_empty()).then(|| commands.join(" && "))
            }
            _ => None,
        }
    }

    fn exit_code(&self) -> Option<i32> {
        let lowered = self.output.to_lowercase();
        let (_, rest) = lowered.rsplit_once("exit code")?;
        let digits: String = rest
            .trim_start_matches([' ', ':'])
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        digits.parse().ok()
    }

    fn denied(&self) -> bool {
        self.output.contains(Self::DENIED)
    }

    fn replacements(diff: &str) -> (String, String) {
        let (mut old, mut new) = (String::new(), String::new());
        let mut side = None;
        for line in diff.lines() {
            let marker = line.trim();
            if marker.ends_with("SEARCH") && marker.starts_with(['-', '<']) {
                side = Some(false);
            } else if marker.starts_with("=======") && side == Some(false) {
                side = Some(true);
            } else if marker.ends_with("REPLACE") && marker.starts_with(['+', '>']) {
                side = None;
            } else if let Some(replacing) = side {
                let target = if replacing { &mut new } else { &mut old };
                target.push_str(line);
                target.push('\n');
            }
        }
        (old, new)
    }

    fn patch_path(patch: &str) -> Option<&str> {
        patch.lines().find_map(|line| {
            ["*** Update File: ", "*** Add File: ", "*** Delete File: "]
                .iter()
                .find_map(|prefix| line.strip_prefix(prefix))
                .map(str::trim)
        })
    }
}

#[derive(Debug)]
pub(crate) struct Cline;

impl Cline {
    const HARNESS: &str = "cline";

    const SHELLS: &[&str] = &["execute_command", "run_commands"];

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "attempt_completion",
        "ask_followup_question",
        "plan_mode_respond",
        "act_mode_respond",
        "focus_chain",
        "todo",
        "new_task",
        "condense",
        "summarize_task",
        "report_bug",
        "load_mcp_documentation",
        "generate_explanation",
    ];

    const EVENTS: &[&str] = &[
        "TaskStart",
        "TaskResume",
        "UserPromptSubmit",
        "PostToolUse",
        "TaskComplete",
        "TaskCancel",
        "SessionShutdown",
    ];

    pub(crate) const MANUAL_STEP: &str = "In VS Code or JetBrains, turn on Cline's \"Enable Hooks\" setting (the Cline CLI needs nothing).";

    fn config() -> Result<PathBuf> {
        Ok(env::home_dir()
            .context("find the home directory")?
            .join(".cline"))
    }

    fn hooks_dir() -> Result<PathBuf> {
        match env::var_os("CLINE_HOOKS_DIR").filter(|dir| !dir.is_empty()) {
            Some(dir) => Ok(PathBuf::from(dir)),
            None => Ok(Self::config()?.join("hooks")),
        }
    }

    fn script(program: &Program) -> String {
        format!(
            "#!/bin/sh\n\
             # Installed by `trodden connect cline`; `trodden disconnect cline` removes it.\n\
             reply=$({})\n\
             if [ -n \"$reply\" ]; then\n  \
               printf '%s\\n' \"$reply\"\n\
             else\n  \
               printf '%s\\n' '{{\"cancel\":false}}'\n\
             fi\n",
            program.hook(Self::HARNESS)
        )
    }

    fn is_ours(path: &Path) -> bool {
        fs::read_to_string(path).is_ok_and(|text| HookFile::is_ours(&text, Self::HARNESS))
    }

    fn others(dir: &Path, event: &str) -> Vec<PathBuf> {
        let Ok(entries) = fs::read_dir(dir) else {
            return Vec::new();
        };
        entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.split('.').next())
                    .is_some_and(|stem| stem.eq_ignore_ascii_case(event))
            })
            .filter(|path| !Self::is_ours(path))
            .collect()
    }

    fn connect_at(dir: &Path, program: &Program) -> Result<Vec<Change>> {
        if cfg!(windows) {
            bail!("Cline runs hook scripts only on macOS and Linux");
        }
        let mut changes = Vec::new();
        for event in Self::EVENTS {
            if let Some(other) = Self::others(dir, event).first() {
                bail!(
                    "{} is a {event} hook Trodden did not write; Cline runs one hook per event, so merge Trodden's by hand: pipe the payload to `{}`",
                    other.display(),
                    program.hook(Self::HARNESS)
                );
            }
            let change = Change::write(dir.join(event), Self::script(program)).executable();
            if change.is_needed() || !Self::is_executable(&change.path) {
                changes.push(change);
            }
        }
        Ok(changes)
    }

    fn is_executable(path: &Path) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::metadata(path).is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
        }
        #[cfg(not(unix))]
        {
            path.exists()
        }
    }

    fn disconnect_at(dir: &Path) -> Vec<Change> {
        Self::EVENTS
            .iter()
            .map(|event| dir.join(event))
            .filter(|path| Self::is_ours(path))
            .map(Change::remove)
            .collect()
    }

    fn connected_at(dir: &Path) -> bool {
        Self::EVENTS
            .iter()
            .all(|event| Self::is_ours(&dir.join(event)))
    }

    fn at(timestamp: Option<&Value>) -> Timestamp {
        let millis = match timestamp {
            Some(Value::Number(number)) => number.as_i64(),
            Some(Value::String(text)) if text.bytes().all(|b| b.is_ascii_digit()) => {
                text.parse().ok()
            }
            Some(Value::String(text)) => return text.parse().unwrap_or_else(|_| Timestamp::now()),
            _ => None,
        };
        millis
            .and_then(|millis| Timestamp::from_millisecond(millis).ok())
            .unwrap_or_else(Timestamp::now)
    }

    fn observe(observer: &Observer<'_>, tool: &ToolUse) -> Option<Observation> {
        let name = tool.name.as_str();
        let succeeded = tool.succeeded && !tool.denied();
        let path = tool.param("path").or_else(|| tool.param("absolutePath"));
        let observation = match name {
            _ if Self::SHELLS.contains(&name) => {
                let command = tool.command()?;
                let exit_code = tool.exit_code();
                observer.command(
                    name,
                    &command,
                    Ran {
                        output: &tool.output,
                        exit_code,
                        failed: !tool.succeeded || exit_code.is_some_and(|code| code != 0),
                        interrupted: tool.denied(),
                        duration_ms: tool.duration_ms,
                    },
                )
            }
            "write_to_file" | "create_file" => {
                let content = tool.param("content").unwrap_or_default();
                observer.edit(name, &path?, Edit::Created { content: &content }, succeeded)
            }
            "replace_in_file" => {
                let (old, new) = ToolUse::replacements(&tool.param("diff").unwrap_or_default());
                observer.edit(
                    name,
                    &path?,
                    Edit::Replaced {
                        old: &old,
                        new: &new,
                    },
                    succeeded,
                )
            }
            "editor" => {
                let text = |key: &str| tool.param(key).unwrap_or_default();
                match tool.param("command").as_deref() {
                    Some("view") => observer.read(name, &path?),
                    Some("create") => observer.edit(
                        name,
                        &path?,
                        Edit::Created {
                            content: &text("file_text"),
                        },
                        succeeded,
                    ),
                    Some("insert") => {
                        let added: String = text("new_str")
                            .lines()
                            .map(|line| format!("+{line}\n"))
                            .collect();
                        observer.edit(name, &path?, Edit::Patched { diff: &added }, succeeded)
                    }
                    _ => observer.edit(
                        name,
                        &path?,
                        Edit::Replaced {
                            old: &text("old_str"),
                            new: &text("new_str"),
                        },
                        succeeded,
                    ),
                }
            }
            "apply_patch" => {
                let patch = tool
                    .param("input")
                    .or_else(|| tool.param("patch"))
                    .unwrap_or_default();
                let path = ToolUse::patch_path(&patch).map(str::to_owned).or(path)?;
                observer.edit(name, &path, Edit::Patched { diff: &patch }, succeeded)
            }
            "delete_file" => observer.edit(name, &path?, Edit::Unknown, succeeded),
            "read_file" | "read_files" => {
                let path = path.or_else(|| tool.param("paths"))?;
                observer.read(name, &path)
            }
            "search_files" | "search_codebase" => observer.search(
                name,
                tool.param("regex")
                    .or_else(|| tool.param("query"))
                    .as_deref(),
                path.as_deref(),
            ),
            "list_files" | "list_code_definition_names" => {
                observer.search(name, None, path.as_deref())
            }
            "web_fetch" | "fetch" => observer.fetch(name, tool.param("url").as_deref(), None),
            "web_search" => observer.fetch(name, None, tool.param("query").as_deref()),
            "use_subagents" | "spawn_agent" => observer.other(name, ToolAction::Delegate),
            _ if Self::BOOKKEEPING_TOOLS.contains(&name) => return None,
            _ => observer.other(name, ToolAction::Other),
        };
        Some(observation)
    }

    fn event_from(input: ClineHookInput, redactor: &Redactor) -> Option<HookEvent> {
        if input.parent_agent_id.is_some() {
            return None;
        }
        let session = input
            .session_context
            .as_ref()
            .and_then(|context| context.root_session_id.clone())
            .or_else(|| input.task_id.clone())
            .filter(|session| !session.is_empty())?;
        let cwd = input
            .workspace_roots
            .first()
            .cloned()
            .or_else(|| {
                input
                    .workspace_info
                    .as_ref()
                    .and_then(|info| info.get("rootPath"))
                    .and_then(Value::as_str)
                    .map(PathBuf::from)
            })
            .filter(|cwd| cwd.is_absolute())?;
        let model = match &input.model {
            Some(Value::String(model)) => Some(model.clone()),
            Some(model) => model
                .get("slug")
                .or_else(|| model.get("id"))
                .and_then(Value::as_str)
                .map(str::to_owned),
            None => None,
        };
        let cwd_text = cwd.to_string_lossy().into_owned();
        let observer = Observer {
            session: &session,
            cwd: &cwd_text,
            at: Self::at(input.timestamp.as_ref()),
            model: model.as_deref(),
            redactor,
        };
        let mut observed = Vec::new();
        let moment = match input.hook_name.as_str() {
            "UserPromptSubmit" | "prompt_submit" => {
                let prompt = input.user_prompt_submit.as_ref()?.prompt.clone();
                observed.push(observer.prompt(&prompt));
                Moment::Prompt(prompt)
            }
            "PostToolUse" | "tool_result" => {
                let tool = ToolUse::of(&input)?;
                observed.extend(Self::observe(&observer, &tool));
                let failed = Self::SHELLS.contains(&tool.name.as_str())
                    && !tool.denied()
                    && (!tool.succeeded || tool.exit_code().is_some_and(|code| code != 0));
                if failed {
                    Moment::CommandFailed(tool.output)
                } else {
                    Moment::ToolDone
                }
            }
            "TaskStart" | "TaskResume" | "agent_start" | "agent_resume" => Moment::SessionStart,
            "TaskComplete" | "agent_end" | "TaskCancel" | "agent_abort" | "TaskError"
            | "agent_error" => Moment::TurnEnd { continued: false },
            "SessionShutdown" | "session_shutdown" => Moment::SessionEnd,
            _ => return None,
        };
        Some(HookEvent {
            name: input.hook_name,
            session,
            cwd,
            transcript: None,
            moment,
            observed,
        })
    }
}

impl Agent for Cline {
    fn title(&self) -> &'static str {
        "Cline"
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
        let input: ClineHookInput =
            serde_json::from_str(payload).context("parse the hook payload")?;
        Ok(Self::event_from(input, redactor))
    }

    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_), Reply::Recall(_)) if event.name != "UserPromptSubmit" => None,
            (Moment::Prompt(_) | Moment::CommandFailed(_), Reply::Recall(envelope)) => {
                Some(json!({ "cancel": false, "contextModification": envelope }).to_string())
            }
            _ => None,
        }
    }

    fn detected(&self) -> bool {
        Self::config().is_ok_and(|config| config.is_dir())
    }

    fn notice(&self) -> Option<&'static str> {
        Some(Self::MANUAL_STEP)
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        Self::connect_at(&Self::hooks_dir()?, program)
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        Ok(Self::disconnect_at(&Self::hooks_dir()?))
    }

    fn connected(&self) -> Result<bool> {
        Ok(Self::connected_at(&Self::hooks_dir()?))
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use trodden_capture::journal::Observed;
    use trodden_core::trace::{EventKind, ToolOutcome};

    use super::*;

    const CWD: &str = "/home/dev/shop";

    #[derive(Debug)]
    struct Payload;

    impl Payload {
        fn vscode(hook: &str, extra: Value) -> Value {
            let mut payload = json!({
                "clineVersion": "4.1.23",
                "hookName": hook,
                "timestamp": "1791540000000",
                "workspaceRoots": [CWD],
                "userId": "u1",
                "taskId": "1791539990000",
                "model": {"provider": "anthropic", "slug": "claude-sonnet-4-5"},
            });
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            payload
        }

        fn event(payload: &Value) -> Option<HookEvent> {
            Cline
                .event(&payload.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
        }

        fn tool(name: &str, parameters: Value, result: &str, success: bool) -> Option<HookEvent> {
            Self::event(&Self::vscode(
                "PostToolUse",
                json!({"postToolUse": {"toolName": name, "parameters": parameters, "result": result,
                    "success": success, "executionTimeMs": 900}}),
            ))
        }

        fn journal(events: &[HookEvent]) -> Trace {
            let text: String = events
                .iter()
                .flat_map(|event| &event.observed)
                .map(|observation| observation.to_line().expect("line encodes"))
                .collect();
            Cline
                .parse(&text, &Redactor::with_home("/home/dev"))
                .expect("journal parses")
                .0
        }
    }

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-cline-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
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
    fn prompts_are_recalled_and_journaled() {
        let event = Payload::event(&Payload::vscode(
            "UserPromptSubmit",
            json!({"userPromptSubmit": {"prompt": "Fix the paging bug in src/paginate.js", "attachments": []}}),
        ))
        .expect("event");

        assert_eq!(event.session, "1791539990000");
        assert_eq!(event.cwd, PathBuf::from(CWD));
        assert_eq!(
            event.moment,
            Moment::Prompt("Fix the paging bug in src/paginate.js".to_owned())
        );
        assert_eq!(event.observed.len(), 1);
        assert_eq!(
            event.observed[0].model.as_deref(),
            Some("claude-sonnet-4-5")
        );
        assert_eq!(
            event.observed[0].at,
            Timestamp::from_millisecond(1_791_540_000_000).expect("valid")
        );
        assert!(
            matches!(&event.observed[0].observed, Observed::Prompt { text } if text.starts_with("Fix the paging"))
        );
    }

    #[test]
    fn shell_failures_are_recalled_and_every_call_journaled() {
        let failed = Payload::tool(
            "execute_command",
            json!({"command": "npm test", "requires_approval": "false"}),
            "Command failed with exit code 1.\nOutput:\nError: Cannot find module 'left-pad'",
            true,
        )
        .expect("event");
        let passed = Payload::tool(
            "execute_command",
            json!({"command": "npm test"}),
            "1 passing",
            true,
        )
        .expect("event");
        let denied = Payload::tool(
            "execute_command",
            json!({"command": "rm -rf dist"}),
            "The user denied this operation.",
            false,
        )
        .expect("event");

        assert!(
            matches!(&failed.moment, Moment::CommandFailed(output) if output.contains("left-pad"))
        );
        assert_eq!(passed.moment, Moment::ToolDone);
        assert_eq!(denied.moment, Moment::ToolDone);
        let trace = Payload::journal(&[failed, passed, denied]);
        let outcomes: Vec<ToolOutcome> = trace
            .events
            .into_iter()
            .filter_map(|event| match event.kind {
                EventKind::ToolCall(call) => Some(call.outcome),
                _ => None,
            })
            .collect();
        assert_eq!(
            outcomes,
            [
                ToolOutcome::Failed { exit_code: Some(1) },
                ToolOutcome::Succeeded,
                ToolOutcome::Interrupted
            ]
        );
    }

    #[test]
    fn sdk_tool_results_are_read_too() {
        let event = Payload::event(&json!({
            "clineVersion": "3.0.70", "hookName": "tool_result", "timestamp": "2026-10-09T10:00:09.000Z",
            "taskId": "run_1", "sessionContext": {"rootSessionId": "1791540000000_k3x9q"},
            "workspaceRoots": [CWD], "userId": "me", "agent_id": "lead", "parent_agent_id": null,
            "tool_result": {"id": "call_1", "name": "run_commands", "input": {"commands": ["npm test"]},
                "output": "1 failing", "error": "Error: expected 2 to equal 3", "durationMs": 5120},
            "postToolUse": {"toolName": "run_commands", "parameters": {"commands": "[\"npm test\"]"},
                "result": "1 failing", "success": false, "executionTimeMs": 5120}
        }))
        .expect("event");

        assert_eq!(event.session, "1791540000000_k3x9q");
        assert!(
            matches!(&event.moment, Moment::CommandFailed(output) if output.contains("expected 2 to equal 3"))
        );
        assert!(matches!(&event.observed[0].observed,
            Observed::Call { args, duration_ms: Some(5120), .. } if args.command.as_deref() == Some("npm test")));
        let subagent = Payload::event(
            &json!({"hookName": "tool_result", "taskId": "t", "workspaceRoots": [CWD],
            "parent_agent_id": "lead", "tool_result": {"name": "run_commands", "input": {"commands": ["ls"]}}}),
        );
        assert_eq!(subagent, None);
    }

    #[test]
    fn edits_reads_and_searches_are_journaled() {
        let events = [
            Payload::tool("read_file", json!({"path": "src/paginate.js"}), "export function paginate", true),
            Payload::tool(
                "replace_in_file",
                json!({"path": "src/paginate.js", "diff": "------- SEARCH\n  const end = start + perPage + 1;\n=======\n  const end = start + perPage;\n+++++++ REPLACE\n"}),
                "The content was successfully saved to src/paginate.js.",
                true,
            ),
            Payload::tool("write_to_file", json!({"path": "test/paging.test.js", "content": "a\nb\n"}), "saved", true),
            Payload::tool("search_files", json!({"path": "src", "regex": "paginate"}), "Found 2 results", true),
            Payload::tool("attempt_completion", json!({"result": "Done"}), "", true),
        ]
        .map(|event| event.expect("event"));

        assert!(events.iter().all(|event| event.moment == Moment::ToolDone));
        assert!(
            events[4].observed.is_empty(),
            "bookkeeping is not journaled"
        );
        let calls: Vec<String> = Payload::journal(&events)
            .events
            .into_iter()
            .filter_map(|event| match event.kind {
                EventKind::ToolCall(call) => Some(format!(
                    "{:?} {} {:?}",
                    call.action,
                    call.args.path.unwrap_or_default(),
                    call.changes
                        .iter()
                        .map(|change| (change.lines_added, change.lines_removed))
                        .collect::<Vec<_>>()
                )),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            [
                "Read src/paginate.js []",
                "Edit src/paginate.js [(1, 1)]",
                "Edit test/paging.test.js [(2, 0)]",
                "Search src []",
            ]
        );
    }

    #[test]
    fn lifecycle_events_map_to_moments() {
        let moment = |hook: &str| {
            Payload::event(&Payload::vscode(hook, json!({}))).map(|event| event.moment)
        };

        assert_eq!(moment("TaskStart"), Some(Moment::SessionStart));
        assert_eq!(moment("agent_resume"), Some(Moment::SessionStart));
        assert_eq!(
            moment("TaskComplete"),
            Some(Moment::TurnEnd { continued: false })
        );
        assert_eq!(
            moment("TaskCancel"),
            Some(Moment::TurnEnd { continued: false })
        );
        assert_eq!(moment("session_shutdown"), Some(Moment::SessionEnd));
        assert_eq!(moment("PreToolUse"), None);
        assert_eq!(
            Payload::event(
                &json!({"hookName": "TaskStart", "taskId": "t", "workspaceRoots": ["shop"]})
            ),
            None
        );
        assert!(Cline.event("[]", &Redactor::new()).is_err());
    }

    #[test]
    fn replies_inject_context_and_never_continue() {
        let event = |moment| HookEvent {
            name: "UserPromptSubmit".to_owned(),
            session: "s".to_owned(),
            cwd: PathBuf::from(CWD),
            transcript: None,
            moment,
            observed: Vec::new(),
        };

        assert_eq!(
            Cline.render(
                &event(Moment::Prompt("p".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some(r#"{"cancel":false,"contextModification":"<m/>"}"#.to_owned())
        );
        assert_eq!(
            Cline.render(
                &event(Moment::CommandFailed("e".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some(r#"{"cancel":false,"contextModification":"<m/>"}"#.to_owned())
        );
        assert_eq!(
            Cline.render(
                &event(Moment::TurnEnd { continued: false }),
                Reply::Remind("run it")
            ),
            None
        );
    }

    #[test]
    fn cli_prompts_are_journaled_but_never_recalled() {
        let event = Payload::event(&json!({
            "clineVersion": "3.0.70", "hookName": "prompt_submit", "timestamp": "2026-10-09T10:00:00.000Z",
            "taskId": "run_1", "sessionContext": {"rootSessionId": "1791540000000_k3x9q"},
            "workspaceRoots": [CWD], "userId": "me", "agent_id": "lead", "parent_agent_id": null,
            "userPromptSubmit": {"prompt": "Fix the paging bug in src/paginate.js"}
        }))
        .expect("event");

        assert_eq!(
            event.moment,
            Moment::Prompt("Fix the paging bug in src/paginate.js".to_owned())
        );
        assert_eq!(event.observed.len(), 1);
        assert_eq!(Cline.render(&event, Reply::Recall("<m/>")), None);
    }

    #[test]
    fn connect_writes_one_script_per_event_and_round_trips() {
        let scratch = Scratch::new("connect");
        fs::write(
            scratch.dir.join("PreToolUse.py"),
            "#!/usr/bin/env python3\n",
        )
        .expect("written");
        let program = Program::at("/home/dev/.cargo/bin/trodden");

        assert!(!Cline::connected_at(&scratch.dir));
        Scratch::apply(Cline::connect_at(&scratch.dir, &program).expect("connect"));

        assert!(Cline::connected_at(&scratch.dir));
        let script = fs::read_to_string(scratch.dir.join("PostToolUse")).expect("script");
        assert!(script.starts_with("#!/bin/sh\n"), "{script}");
        assert!(
            script.contains("reply=$(/home/dev/.cargo/bin/trodden hook cline)"),
            "{script}"
        );
        assert!(
            script.contains(r#"printf '%s\n' '{"cancel":false}'"#),
            "{script}"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = fs::metadata(scratch.dir.join("PostToolUse"))
                .expect("stat")
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o755);
        }
        assert!(
            Cline::connect_at(&scratch.dir, &program)
                .expect("connect")
                .is_empty()
        );

        Scratch::apply(Cline::disconnect_at(&scratch.dir));
        assert!(!Cline::connected_at(&scratch.dir));
        assert!(!scratch.dir.join("PostToolUse").exists());
        assert!(
            scratch.dir.join("PreToolUse.py").exists(),
            "the user's hooks stay"
        );
        assert!(!Cline::MANUAL_STEP.is_empty());
    }

    #[test]
    fn a_users_own_hook_for_the_same_event_is_not_overwritten() {
        let scratch = Scratch::new("conflict");
        fs::write(scratch.dir.join("TaskComplete.sh"), "#!/bin/sh\nsay done\n").expect("written");

        let error = Cline::connect_at(&scratch.dir, &Program::at("/opt/trodden"))
            .expect_err("a foreign hook blocks connect");

        assert!(
            format!("{error:#}").contains("TaskComplete.sh"),
            "{error:#}"
        );
    }
}
