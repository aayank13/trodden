use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use jiff::Timestamp;
use serde::Deserialize;
use serde_json::{Value, json};
use trodden_capture::{
    journal::{Edit, Journal, Observation, Observer, Ran},
    kimi::HARNESS,
};
use trodden_core::{Trace, trace::ToolAction};
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply};
use crate::connect::{HookFile, Program};

#[derive(Debug)]
pub(crate) struct Kimi;

#[derive(Debug, Deserialize)]
struct KimiHookInput {
    hook_event_name: String,
    session_id: String,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default)]
    prompt: Option<Value>,
    #[serde(default)]
    is_steer: bool,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_input: Option<Value>,
    #[serde(default)]
    tool_output: Option<String>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default)]
    trigger: Option<String>,
}

impl Kimi {
    const SHELL: &str = "Bash";
}

impl Agent for Kimi {
    fn title(&self) -> &'static str {
        "Kimi Code"
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
            Journal::parse(text, HARNESS, redactor)?,
            Journal::working_directory(text).map(PathBuf::from),
        ))
    }

    fn event(&self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>> {
        let input: KimiHookInput =
            serde_json::from_str(payload).context("parse the Kimi Code hook payload")?;
        let Some(cwd) = input.cwd.clone().filter(|cwd| cwd.is_absolute()) else {
            return Ok(None);
        };
        if input.session_id.is_empty() {
            return Ok(None);
        }
        let cwd_text = cwd.to_string_lossy().into_owned();
        let observer = Observer {
            session: &input.session_id,
            cwd: &cwd_text,
            at: Timestamp::now(),
            model: None,
            redactor,
        };
        let (moment, observed) = match input.hook_event_name.as_str() {
            "SessionStart" => (Moment::SessionStart, Vec::new()),
            "UserPromptSubmit" => {
                let Some(prompt) = input.prompt_text().filter(|_| !input.is_steer) else {
                    return Ok(None);
                };
                let observed = vec![observer.prompt(&prompt)];
                (Moment::Prompt(prompt), observed)
            }
            "PostToolUse" => match input.observe(&observer, None) {
                Some(observation) => (Moment::ToolDone, vec![observation]),
                None => return Ok(None),
            },
            "PostToolUseFailure" => {
                let error = input.error_text();
                let Some(observation) = input.observe(&observer, Some(&error)) else {
                    return Ok(None);
                };
                let moment = if input.tool_name.as_deref() == Some(Self::SHELL)
                    && !KimiHookInput::interrupted(&error)
                {
                    Moment::CommandFailed(error)
                } else {
                    Moment::ToolDone
                };
                (moment, vec![observation])
            }
            "Stop" => (Moment::TurnEnd { continued: false }, Vec::new()),
            "PreCompact" => {
                let automatic = input.trigger.as_deref() != Some("manual");
                (Moment::Compacting, vec![observer.compaction(automatic)])
            }
            "SessionEnd" => (Moment::SessionEnd, Vec::new()),
            _ => return Ok(None),
        };
        Ok(Some(HookEvent {
            name: input.hook_event_name,
            session: input.session_id,
            cwd,
            transcript: None,
            moment,
            observed,
        }))
    }

    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_), Reply::Recall(envelope)) => Some(envelope.to_owned()),
            (Moment::TurnEnd { .. }, Reply::Remind(reason)) => Some(
                json!({
                    "hookSpecificOutput": {
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                })
                .to_string(),
            ),
            _ => None,
        }
    }

    fn detected(&self) -> bool {
        KimiHome::locate().is_ok_and(|home| home.dir.is_dir())
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        KimiHome::locate()?.connect(program)
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        KimiHome::locate()?.disconnect()
    }

    fn connected(&self) -> Result<bool> {
        KimiHome::locate()?.connected()
    }
}

impl KimiHookInput {
    fn prompt_text(&self) -> Option<String> {
        let text = match self.prompt.as_ref()? {
            Value::String(text) => text.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter(|part| {
                    part.get("type")
                        .and_then(Value::as_str)
                        .is_none_or(|kind| kind == "text")
                })
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => return None,
        };
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_owned())
    }

    fn error_text(&self) -> String {
        match &self.error {
            Some(Value::String(text)) => text.clone(),
            Some(error) => error
                .get("message")
                .and_then(Value::as_str)
                .map_or_else(|| error.to_string(), str::to_owned),
            None => String::new(),
        }
    }

    fn exit_code(error: &str) -> Option<i32> {
        let rest = error.rsplit_once("Command failed with exit code: ")?.1;
        let digits: String = rest
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        digits.parse().ok()
    }

    fn interrupted(error: &str) -> bool {
        let error = error.trim_end();
        error.ends_with("Interrupted by user")
            || error.starts_with("Aborted before command started")
            || error.contains("Command killed by timeout")
            || error.contains("was denied by permission policy")
            || error.contains("rejected by the user")
    }

    fn observe(&self, observer: &Observer<'_>, error: Option<&str>) -> Option<Observation> {
        let tool = self.tool_name.as_deref()?;
        let input = self.tool_input.clone().unwrap_or_default();
        let field = |key: &str| input.get(key).and_then(Value::as_str);
        let succeeded = error.is_none();
        let observation = match tool {
            "Bash" => {
                let command = field("command")?;
                let output = error.or(self.tool_output.as_deref()).unwrap_or_default();
                let ran = Ran {
                    output,
                    exit_code: error.and_then(Self::exit_code),
                    failed: !succeeded,
                    interrupted: error.is_some_and(Self::interrupted),
                    duration_ms: None,
                };
                match field("cwd").filter(|dir| Path::new(dir).is_absolute()) {
                    Some(dir) => Observer {
                        session: observer.session,
                        cwd: dir,
                        at: observer.at,
                        model: observer.model,
                        redactor: observer.redactor,
                    }
                    .command(tool, command, ran),
                    None => observer.command(tool, command, ran),
                }
            }
            "Edit" => observer.edit(
                tool,
                field("path")?,
                Edit::Replaced {
                    old: field("old_string").unwrap_or_default(),
                    new: field("new_string").unwrap_or_default(),
                },
                succeeded,
            ),
            "Write" => {
                let content = field("content").unwrap_or_default();
                let edit = if field("mode") == Some("append") {
                    Edit::Replaced {
                        old: "",
                        new: content,
                    }
                } else {
                    Edit::Created { content }
                };
                observer.edit(tool, field("path")?, edit, succeeded)
            }
            "Read" => observer.read(tool, field("path")?),
            "Grep" | "Glob" => observer.search(tool, field("pattern"), field("path")),
            "FetchURL" => observer.fetch(tool, field("url"), None),
            "WebSearch" => observer.fetch(tool, None, field("query")),
            "Agent" => observer.other(tool, ToolAction::Delegate),
            _ => return None,
        };
        Some(observation)
    }
}

#[derive(Debug, Clone)]
struct KimiHome {
    dir: PathBuf,
}

impl KimiHome {
    const TOOLS: &str = "^(Bash|Edit|Write|Read|Grep|Glob|FetchURL|WebSearch|Agent)$";

    const EVENTS: &[(&str, Option<&str>)] = &[
        ("SessionStart", None),
        ("UserPromptSubmit", None),
        ("PostToolUse", Some(Self::TOOLS)),
        ("PostToolUseFailure", Some(Self::TOOLS)),
        ("Stop", None),
        ("PreCompact", None),
        ("SessionEnd", None),
    ];

    const TIMEOUT_SECONDS: u64 = 5;

    fn locate() -> Result<Self> {
        let dir = match env::var_os("KIMI_CODE_HOME").filter(|dir| !dir.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => env::home_dir()
                .context("find the home directory")?
                .join(".kimi-code"),
        };
        Ok(Self { dir })
    }

    fn config(&self) -> PathBuf {
        self.dir.join("config.toml")
    }

    fn read(&self) -> Result<Option<String>> {
        let path = self.config();
        match fs::read_to_string(&path) {
            Ok(text) => Ok(Some(text)),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
        }
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        let original = self.read()?;
        let text = original.as_deref().unwrap_or_default();
        if TomlHooks::inline(text) {
            bail!(
                "{} sets `hooks` inline; add Trodden's hooks there by hand",
                self.config().display()
            );
        }
        let mut updated = TomlHooks::without_ours(text);
        let command = TomlHooks::quoted(&program.hook("kimi"));
        if !updated.is_empty() {
            if !updated.ends_with('\n') {
                updated.push('\n');
            }
            if !updated.ends_with("\n\n") {
                updated.push('\n');
            }
        }
        let blocks: Vec<String> = Self::EVENTS
            .iter()
            .map(|(event, matcher)| {
                let matcher = matcher
                    .map(|matcher| format!("matcher = {}\n", TomlHooks::quoted(matcher)))
                    .unwrap_or_default();
                format!(
                    "[[hooks]]\nevent = \"{event}\"\n{matcher}command = {command}\ntimeout = {}\n",
                    Self::TIMEOUT_SECONDS
                )
            })
            .collect();
        updated.push_str(&blocks.join("\n"));
        Ok(self.change(original.as_deref(), updated))
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        let Some(original) = self.read()? else {
            return Ok(Vec::new());
        };
        let updated = TomlHooks::without_ours(&original);
        Ok(self.change(Some(&original), updated))
    }

    fn connected(&self) -> Result<bool> {
        Ok(self
            .read()?
            .is_some_and(|text| TomlHooks::contains_ours(&text)))
    }

    fn change(&self, original: Option<&str>, updated: String) -> Vec<Change> {
        if original == Some(updated.as_str()) || (original.is_none() && updated.is_empty()) {
            return Vec::new();
        }
        vec![Change::write(self.config(), updated)]
    }
}

// No TOML parser: Trodden's `[[hooks]]` tables are found and removed line by line.
#[derive(Debug)]
struct TomlHooks;

impl TomlHooks {
    fn is_header(line: &str) -> bool {
        line.trim_start().starts_with('[')
    }

    fn is_hooks_header(line: &str) -> bool {
        line.trim_start()
            .strip_prefix("[[")
            .and_then(|rest| rest.trim_start().strip_prefix("hooks"))
            .is_some_and(|rest| rest.trim_start().starts_with("]]"))
    }

    fn inline(text: &str) -> bool {
        text.lines()
            .take_while(|line| !Self::is_header(line))
            .any(|line| {
                line.trim_start()
                    .strip_prefix("hooks")
                    .is_some_and(|rest| rest.trim_start().starts_with('='))
            })
    }

    fn tables(text: &str) -> Vec<Vec<&str>> {
        let mut tables: Vec<Vec<&str>> = vec![Vec::new()];
        for line in text.split_inclusive('\n') {
            if Self::is_header(line) {
                tables.push(Vec::new());
            }
            if let Some(table) = tables.last_mut() {
                table.push(line);
            }
        }
        tables
    }

    fn is_ours(table: &[&str]) -> bool {
        table
            .first()
            .is_some_and(|header| Self::is_hooks_header(header))
            && table.iter().any(|line| {
                line.trim_start()
                    .strip_prefix("command")
                    .is_some_and(|rest| rest.trim_start().starts_with('='))
                    && HookFile::is_ours(line, "kimi")
            })
    }

    fn contains_ours(text: &str) -> bool {
        Self::tables(text).iter().any(|table| Self::is_ours(table))
    }

    fn without_ours(text: &str) -> String {
        let tables = Self::tables(text);
        if !tables.iter().any(|table| Self::is_ours(table)) {
            return text.to_owned();
        }
        let kept: String = tables
            .iter()
            .filter(|table| !Self::is_ours(table))
            .flatten()
            .copied()
            .collect();
        Self::tidy(&kept)
    }

    fn tidy(text: &str) -> String {
        let mut tidy = String::with_capacity(text.len());
        let mut blank = 0;
        for line in text.split_inclusive('\n') {
            if line.trim().is_empty() {
                blank += 1;
                if blank > 1 {
                    continue;
                }
            } else {
                blank = 0;
            }
            tidy.push_str(line);
        }
        let trimmed = tidy.trim_end();
        if trimmed.is_empty() {
            String::new()
        } else {
            format!("{trimmed}\n")
        }
    }

    fn quoted(text: &str) -> String {
        let mut quoted = String::with_capacity(text.len() + 2);
        quoted.push('"');
        for c in text.chars() {
            match c {
                '"' => quoted.push_str("\\\""),
                '\\' => quoted.push_str("\\\\"),
                '\n' => quoted.push_str("\\n"),
                '\t' => quoted.push_str("\\t"),
                c if c.is_control() => quoted.push_str(&format!("\\u{:04X}", u32::from(c))),
                c => quoted.push(c),
            }
        }
        quoted.push('"');
        quoted
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use trodden_core::trace::{EventKind, ToolOutcome};

    use super::*;

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-kimi-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn home(&self) -> KimiHome {
            KimiHome {
                dir: self.dir.join(".kimi-code"),
            }
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

    #[derive(Debug)]
    struct Payload;

    impl Payload {
        fn event(payload: Value) -> Option<HookEvent> {
            let mut base = json!({"session_id": "session_abc", "session_title": "Fix paging", "client_type": "kimi_code_cli", "cwd": "/home/dev/shop"});
            base.as_object_mut()
                .expect("object")
                .extend(payload.as_object().expect("object").clone());
            Kimi.event(&base.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
        }

        fn journal(payloads: &[Value]) -> Trace {
            let mut text = String::new();
            for payload in payloads {
                for observation in Self::event(payload.clone()).expect("event").observed {
                    text.push_str(&observation.to_line().expect("line"));
                }
            }
            Kimi.parse(&text, &Redactor::with_home("/home/dev"))
                .expect("journal parses")
                .0
        }
    }

    #[test]
    fn prompts_recall_from_text_or_content_parts() {
        let event = Payload::event(json!({"hook_event_name": "UserPromptSubmit", "prompt": [{"type": "text", "text": "fix the paging bug"}], "is_steer": false}))
            .expect("event");
        assert_eq!(
            event.moment,
            Moment::Prompt("fix the paging bug".to_owned())
        );
        assert_eq!(event.observed.len(), 1);
        assert_eq!(event.transcript, None);

        assert_eq!(
            Payload::event(json!({"hook_event_name": "UserPromptSubmit", "prompt": "fix it"}))
                .map(|e| e.moment),
            Some(Moment::Prompt("fix it".to_owned()))
        );
        assert!(Payload::event(json!({"hook_event_name": "UserPromptSubmit", "prompt": "also this", "is_steer": true})).is_none());
        assert!(
            Payload::event(
                json!({"hook_event_name": "UserPromptSubmit", "prompt": [{"type": "image_url"}]})
            )
            .is_none()
        );
    }

    #[test]
    fn tool_events_are_journaled() {
        let failed = Payload::event(json!({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "tool_input": {"command": "npm test"}, "tool_call_id": "c1",
            "error": {"code": "internal", "message": "1 failing\nCommand failed with exit code: 1.", "retryable": false}}))
            .expect("event");
        assert_eq!(
            failed.moment,
            Moment::CommandFailed("1 failing\nCommand failed with exit code: 1.".to_owned())
        );
        assert_eq!(failed.observed.len(), 1);

        let interrupted = Payload::event(json!({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "tool_input": {"command": "sleep 9"},
            "error": {"message": "Interrupted by user"}}))
            .expect("event");
        assert_eq!(interrupted.moment, Moment::ToolDone);

        let done = Payload::event(json!({"hook_event_name": "PostToolUse", "tool_name": "Read", "tool_input": {"path": "src/a.js"}, "tool_output": "x"}))
            .expect("event");
        assert_eq!(done.moment, Moment::ToolDone);
        assert!(
            Payload::event(
                json!({"hook_event_name": "PostToolUse", "tool_name": "mcp__x", "tool_input": {}})
            )
            .is_none()
        );
        assert!(
            Payload::event(
                json!({"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {}})
            )
            .is_none()
        );

        assert_eq!(
            Payload::event(json!({"hook_event_name": "Stop", "stop_hook_active": false}))
                .map(|e| e.moment),
            Some(Moment::TurnEnd { continued: false })
        );
        let compacting = Payload::event(
            json!({"hook_event_name": "PreCompact", "trigger": "auto", "token_count": 9}),
        )
        .expect("event");
        assert_eq!(
            (compacting.moment, compacting.observed.len()),
            (Moment::Compacting, 1)
        );
        assert_eq!(
            Payload::event(json!({"hook_event_name": "SessionEnd", "reason": "exit"}))
                .map(|e| e.moment),
            Some(Moment::SessionEnd)
        );
        assert_eq!(
            Payload::event(json!({"hook_event_name": "SessionStart", "source": "startup"}))
                .map(|e| e.moment),
            Some(Moment::SessionStart)
        );
        assert!(Payload::event(json!({"hook_event_name": "Notification"})).is_none());
        assert!(
            Kimi.event(
                &json!({"hook_event_name": "Stop", "session_id": "s", "cwd": "relative"})
                    .to_string(),
                &Redactor::with_home("/home/dev")
            )
            .expect("parses")
            .is_none()
        );
    }

    #[test]
    fn the_journal_becomes_a_trace() {
        let trace = Payload::journal(&[
            json!({"hook_event_name": "UserPromptSubmit", "prompt": [{"type": "text", "text": "fix the paging bug"}]}),
            json!({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "tool_input": {"command": "npm test"},
                   "error": {"message": "AssertionError: expected 3 to equal 2\nCommand failed with exit code: 1."}}),
            json!({"hook_event_name": "PostToolUse", "tool_name": "Edit", "tool_input": {"path": "/home/dev/shop/src/paginate.js", "old_string": "a\n+ 1", "new_string": "a\n"}, "tool_output": "ok"}),
            json!({"hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_input": {"command": "npm test", "cwd": "/home/dev/shop/web"}, "tool_output": "3 passing"}),
            json!({"hook_event_name": "PreCompact", "trigger": "manual"}),
        ]);
        let kinds: Vec<String> = trace
            .events
            .iter()
            .map(|event| match &event.kind {
                EventKind::Prompt { summary } => format!("prompt {summary}"),
                EventKind::ToolCall(call) => format!(
                    "{} {:?} {:?} {:?}",
                    call.tool, call.args.command, call.args.path, call.outcome
                ),
                EventKind::Compaction { automatic } => format!("compaction {automatic}"),
                _ => "other".to_owned(),
            })
            .collect();

        assert_eq!(trace.session.as_str(), "session_abc");
        assert_eq!(trace.harness.as_str(), "kimi");
        assert_eq!(
            kinds,
            [
                "prompt fix the paging bug".to_owned(),
                format!(
                    "Bash Some(\"npm test\") None {:?}",
                    ToolOutcome::Failed { exit_code: Some(1) }
                ),
                format!(
                    "Edit None Some(\"src/paginate.js\") {:?}",
                    ToolOutcome::Succeeded
                ),
                format!(
                    "Bash Some(\"cd web && npm test\") None {:?}",
                    ToolOutcome::Succeeded
                ),
                "compaction false".to_owned(),
            ]
        );
        let EventKind::ToolCall(edit) = &trace.events[2].kind else {
            panic!("an edit");
        };
        assert_eq!(
            (edit.changes[0].lines_added, edit.changes[0].lines_removed),
            (0, 1)
        );
    }

    #[test]
    fn replies_take_the_shape_kimi_reads() {
        let event = |moment| HookEvent {
            name: String::new(),
            session: "s".to_owned(),
            cwd: PathBuf::from("/w"),
            transcript: None,
            moment,
            observed: Vec::new(),
        };

        assert_eq!(
            Kimi.render(
                &event(Moment::Prompt("p".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some("<m/>".to_owned())
        );
        assert_eq!(
            Kimi.render(
                &event(Moment::CommandFailed("e".to_owned())),
                Reply::Recall("<m/>")
            ),
            None
        );
        let stop: Value = serde_json::from_str(
            &Kimi
                .render(
                    &event(Moment::TurnEnd { continued: false }),
                    Reply::Remind("run it"),
                )
                .expect("renders"),
        )
        .expect("JSON");
        assert_eq!(stop["hookSpecificOutput"]["permissionDecision"], "deny");
        assert_eq!(
            stop["hookSpecificOutput"]["permissionDecisionReason"],
            "run it"
        );
    }

    #[test]
    fn connect_round_trips_and_keeps_other_tables() {
        let scratch = Scratch::new("connect");
        let home = scratch.home();
        fs::create_dir_all(&home.dir).expect("dir");
        let user = "default_model = \"kimi-k2\"\n\n[[hooks]]\nevent = \"Notification\"\nmatcher = \"task\\\\.completed\"\ncommand = \"terminal-notifier -message done\"\n\n[mcp]\nstartup_timeout_ms = 3000\n";
        fs::write(home.config(), user).expect("written");
        let program = Program::at("/opt/bin/trodden");

        assert!(!home.connected().expect("config reads"));
        Scratch::apply(home.connect(&program).expect("connect"));
        assert!(home.connected().expect("config reads"));
        assert!(home.connect(&program).expect("connect again").is_empty());
        let text = fs::read_to_string(home.config()).expect("read");
        assert!(text.starts_with(user), "{text}");
        assert!(text.contains("[[hooks]]\nevent = \"PostToolUse\"\nmatcher = \"^(Bash|Edit|Write|Read|Grep|Glob|FetchURL|WebSearch|Agent)$\"\ncommand = \"/opt/bin/trodden hook kimi\"\ntimeout = 5\n"), "{text}");
        assert_eq!(text.matches("[[hooks]]").count(), 8);

        Scratch::apply(home.disconnect().expect("disconnect"));
        assert!(!home.connected().expect("config reads"));
        assert_eq!(fs::read_to_string(home.config()).expect("read"), user);
    }

    #[test]
    fn a_missing_config_is_created_and_inline_hooks_are_refused() {
        let scratch = Scratch::new("fresh");
        let home = scratch.home();

        assert!(home.disconnect().expect("nothing to remove").is_empty());
        let changes = home
            .connect(&Program::at("/Users/dev/Application Support/trodden"))
            .expect("connect");
        let text = changes[0].contents.clone().expect("contents");
        assert!(text.starts_with("[[hooks]]\nevent = \"SessionStart\"\ncommand = \"'/Users/dev/Application Support/trodden' hook kimi\"\n"), "{text}");

        fs::create_dir_all(&home.dir).expect("dir");
        fs::write(home.config(), "hooks = []\n").expect("written");
        let error = home
            .connect(&Program::at("/opt/bin/trodden"))
            .expect_err("inline hooks are refused");
        assert!(format!("{error:#}").contains("by hand"));
    }

    #[test]
    fn toml_strings_are_escaped() {
        assert_eq!(
            TomlHooks::quoted(r#"C:\bin\trodden "x""#),
            r#""C:\\bin\\trodden \"x\"""#
        );
        assert!(TomlHooks::is_hooks_header("  [[ hooks ]] # ours\n"));
        assert!(!TomlHooks::is_hooks_header("[hooks_extra]\n"));
    }
}
