use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use trodden_capture::copilot::Events;
use trodden_core::Trace;
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply, claude_code::ClaudeHookInput};
use crate::connect::{HookFile, Program};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct CopilotHookInput {
    #[serde(alias = "session_id")]
    session_id: String,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default, alias = "hook_event_name")]
    hook_event_name: Option<String>,
    #[serde(default, alias = "transcript_path")]
    transcript_path: Option<PathBuf>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default, alias = "transformed_prompt")]
    transformed_prompt: Option<String>,
    #[serde(default, alias = "tool_name")]
    tool_name: Option<String>,
    #[serde(default, alias = "tool_input")]
    tool_args: Option<Value>,
    #[serde(default, alias = "tool_result")]
    tool_result: Option<Value>,
    #[serde(default)]
    error: Option<Value>,
    #[serde(default, alias = "stop_reason")]
    stop_reason: Option<String>,
    #[serde(default, rename = "stop_hook_active", alias = "stopHookActive")]
    stop_hook_active: bool,
    #[serde(default)]
    trigger: Option<String>,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    source: Option<String>,
    #[serde(default, alias = "agent_name")]
    agent_name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CopilotEvent {
    SessionStart,
    Prompt,
    ToolDone,
    ToolFailed,
    Stop,
    PreCompact,
    SessionEnd,
}

impl CopilotHookInput {
    fn kind(&self) -> Option<CopilotEvent> {
        if let Some(name) = &self.hook_event_name {
            return match name.as_str() {
                "sessionStart" | "SessionStart" => Some(CopilotEvent::SessionStart),
                "userPromptSubmitted" | "UserPromptSubmit" => Some(CopilotEvent::Prompt),
                "postToolUse" | "PostToolUse" => Some(CopilotEvent::ToolDone),
                "postToolUseFailure" | "PostToolUseFailure" => Some(CopilotEvent::ToolFailed),
                "agentStop" | "Stop" => Some(CopilotEvent::Stop),
                "preCompact" | "PreCompact" => Some(CopilotEvent::PreCompact),
                "sessionEnd" | "SessionEnd" => Some(CopilotEvent::SessionEnd),
                _ => None,
            };
        }
        if self.agent_name.is_some() || self.transformed_prompt.is_some() {
            return None;
        }
        if self.tool_name.is_some() {
            return if self.tool_result.is_some() {
                Some(CopilotEvent::ToolDone)
            } else if self.error.is_some() {
                Some(CopilotEvent::ToolFailed)
            } else {
                None
            };
        }
        if self.prompt.is_some() {
            Some(CopilotEvent::Prompt)
        } else if self.stop_reason.is_some() {
            Some(CopilotEvent::Stop)
        } else if self.trigger.is_some() {
            Some(CopilotEvent::PreCompact)
        } else if self.reason.is_some() {
            Some(CopilotEvent::SessionEnd)
        } else if self.source.is_some() {
            Some(CopilotEvent::SessionStart)
        } else {
            None
        }
    }

    fn is_shell(&self) -> bool {
        self.tool_name
            .as_deref()
            .is_some_and(|tool| Copilot::SHELLS.contains(&tool))
    }

    fn command(&self) -> String {
        let args = match &self.tool_args {
            Some(Value::String(text)) => serde_json::from_str(text).unwrap_or_default(),
            Some(value) => value.clone(),
            None => Value::Null,
        };
        args.get("command")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned()
    }

    fn failure(&self, kind: CopilotEvent, redactor: &Redactor) -> Option<String> {
        if !self.is_shell() {
            return None;
        }
        match kind {
            CopilotEvent::ToolDone => {
                let result = self.tool_result.as_ref()?;
                let output = ["textResultForLlm", "text_result_for_llm"]
                    .iter()
                    .find_map(|key| result.get(*key).and_then(Value::as_str))?;
                let kind = ["resultType", "result_type"]
                    .iter()
                    .find_map(|key| result.get(*key).and_then(Value::as_str));
                (kind == Some("failure") || Events::failed(&self.command(), output, redactor))
                    .then(|| output.to_owned())
            }
            CopilotEvent::ToolFailed => {
                let error = match self.error.as_ref()? {
                    Value::String(text) => text.clone(),
                    other => other.get("message")?.as_str()?.to_owned(),
                };
                let lowered = error.to_lowercase();
                let declined = [
                    "rejected",
                    "denied by the user",
                    "user denied",
                    "cancelled",
                    "canceled",
                    "aborted",
                ]
                .iter()
                .any(|phrase| lowered.contains(phrase));
                (!declined && !error.trim().is_empty()).then_some(error)
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Copilot;

impl Copilot {
    const HARNESS: &str = "copilot";

    const SHELLS: &[&str] = &["bash", "powershell", "Bash", "PowerShell"];

    const TIMEOUT_SECONDS: u64 = 5;

    const EVENTS: &[(&str, Option<&str>)] = &[
        ("sessionStart", None),
        ("userPromptSubmitted", None),
        ("postToolUse", Some("bash|powershell")),
        ("postToolUseFailure", None),
        ("agentStop", None),
        ("preCompact", None),
        ("sessionEnd", None),
    ];

    fn config() -> Result<PathBuf> {
        match env::var_os("COPILOT_HOME").filter(|home| !home.is_empty()) {
            Some(home) => Ok(PathBuf::from(home)),
            None => Ok(env::home_dir()
                .context("find the home directory")?
                .join(".copilot")),
        }
    }

    fn hooks(config: &Path) -> PathBuf {
        config.join("hooks").join("trodden.json")
    }

    fn connect_at(config: &Path, program: &Program) -> Result<Vec<Change>> {
        let command = program.hook(Self::HARNESS);
        let mut hooks = serde_json::Map::new();
        for (event, matcher) in Self::EVENTS {
            let mut handler = json!({
                "type": "command",
                "bash": command,
                "powershell": format!("& {command}"),
                "timeoutSec": Self::TIMEOUT_SECONDS,
            });
            if let Some(matcher) = matcher {
                handler["matcher"] = json!(matcher);
            }
            hooks.insert((*event).to_owned(), json!([handler]));
        }
        let mut text = serde_json::to_string_pretty(&json!({"version": 1, "hooks": hooks}))
            .context("encode the Copilot hooks")?;
        text.push('\n');
        let change = Change::write(Self::hooks(config), text);
        Ok(if change.is_needed() {
            vec![change]
        } else {
            Vec::new()
        })
    }

    fn disconnect_at(config: &Path) -> Result<Vec<Change>> {
        Ok(if Self::connected_at(config)? {
            vec![Change::remove(Self::hooks(config))]
        } else {
            Vec::new()
        })
    }

    fn connected_at(config: &Path) -> Result<bool> {
        let path = Self::hooks(config);
        Ok(path.is_file() && HookFile::load(path)?.contains(&["hooks"], Self::HARNESS))
    }

    fn transcript_of(config: &Path, session: &str) -> Option<PathBuf> {
        let plain = !session.is_empty()
            && session
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "-_".contains(c));
        let path = config
            .join("session-state")
            .join(session)
            .join("events.jsonl");
        (plain && path.is_file()).then_some(path)
    }

    fn transcripts_in(sessions: &Path) -> Result<Vec<PathBuf>> {
        let mut transcripts = Vec::new();
        for entry in
            fs::read_dir(sessions).with_context(|| format!("list {}", sessions.display()))?
        {
            let path = entry
                .context("read a session directory entry")?
                .path()
                .join("events.jsonl");
            if path.is_file() {
                transcripts.push(path);
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }

    fn event_at(
        config: Option<&Path>,
        payload: &str,
        redactor: &Redactor,
    ) -> Result<Option<HookEvent>> {
        let input: CopilotHookInput =
            serde_json::from_str(payload).context("parse the hook payload")?;
        let Some(kind) = input.kind() else {
            return Ok(None);
        };
        let moment = match kind {
            CopilotEvent::SessionStart => Moment::SessionStart,
            CopilotEvent::Prompt => match &input.prompt {
                Some(prompt) => Moment::Prompt(prompt.clone()),
                None => return Ok(None),
            },
            CopilotEvent::ToolDone | CopilotEvent::ToolFailed => {
                match input.failure(kind, redactor) {
                    Some(error) => Moment::CommandFailed(error),
                    None => return Ok(None),
                }
            }
            CopilotEvent::Stop => Moment::TurnEnd {
                continued: input.stop_hook_active,
            },
            CopilotEvent::PreCompact => Moment::Compacting,
            CopilotEvent::SessionEnd => Moment::SessionEnd,
        };
        let Some(cwd) = input.cwd.filter(|cwd| cwd.is_absolute()) else {
            return Ok(None);
        };
        let transcript = input
            .transcript_path
            .filter(|path| !path.as_os_str().is_empty())
            .or_else(|| config.and_then(|config| Self::transcript_of(config, &input.session_id)));
        Ok(Some(HookEvent {
            name: input.hook_event_name.unwrap_or_else(|| format!("{kind:?}")),
            session: input.session_id,
            cwd,
            transcript,
            moment,
            observed: Vec::new(),
        }))
    }
}

impl Agent for Copilot {
    fn title(&self) -> &'static str {
        "GitHub Copilot CLI"
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(Some(Self::config()?.join("session-state")))
    }

    fn transcripts(&self, history: &Path) -> Result<Vec<PathBuf>> {
        Self::transcripts_in(history)
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        Ok((
            Events::parse(text, redactor)?,
            Events::working_directory(text),
        ))
    }

    fn event(&self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>> {
        Self::event_at(Self::config().ok().as_deref(), payload, redactor)
    }

    // UNVERIFIED: the docs say `userPromptSubmitted` output is dropped, the changelog says
    // `additionalContext` is honored.
    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_) | Moment::CommandFailed(_), Reply::Recall(envelope)) => {
                Some(json!({ "additionalContext": envelope }).to_string())
            }
            (Moment::TurnEnd { .. }, Reply::Remind(reason)) => Some(ClaudeHookInput::block(reason)),
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
    use std::process;

    use super::*;

    const SESSION: &str = "3f2c9a4e-8b1d-4c6e-9a7f-1d2e3f4a5b6c";

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-copilot-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }

        fn event(&self, payload: Value) -> Option<HookEvent> {
            Copilot::event_at(
                Some(&self.dir),
                &payload.to_string(),
                &Redactor::with_home("/home/dev"),
            )
            .expect("payload parses")
        }

        fn moment(&self, extra: Value) -> Option<Moment> {
            let mut payload = json!({"sessionId": SESSION, "timestamp": 1_791_540_000_000_u64, "cwd": "/home/dev/shop"});
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            self.event(payload).map(|event| event.moment)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn camel_case_events_are_told_apart_by_their_fields() {
        let scratch = Scratch::new("events");

        assert_eq!(
            scratch.moment(json!({"source": "startup", "initialPrompt": "hi"})),
            Some(Moment::SessionStart)
        );
        assert_eq!(
            scratch.moment(json!({"prompt": "Fix paging"})),
            Some(Moment::Prompt("Fix paging".to_owned()))
        );
        assert_eq!(
            scratch.moment(json!({"transcriptPath": "/t/events.jsonl", "stopReason": "end_turn", "stop_hook_active": true})),
            Some(Moment::TurnEnd { continued: true })
        );
        assert_eq!(
            scratch.moment(json!({"transcriptPath": "/t/events.jsonl", "trigger": "auto", "customInstructions": ""})),
            Some(Moment::Compacting)
        );
        assert_eq!(
            scratch.moment(json!({"reason": "user_exit"})),
            Some(Moment::SessionEnd)
        );
        assert_eq!(
            scratch.moment(json!({"prompt": "p", "transformedPrompt": "<x/>p"})),
            None
        );
        assert_eq!(
            scratch.moment(
                json!({"transcriptPath": "/t", "agentName": "explore", "stopReason": "end_turn"})
            ),
            None
        );
        assert_eq!(
            scratch.moment(json!({"toolName": "bash", "toolArgs": {"command": "npm test"}})),
            None
        );
        assert_eq!(scratch.moment(json!({})), None);
    }

    #[test]
    fn pascal_case_events_carry_their_name() {
        let scratch = Scratch::new("pascal");
        let event = scratch
            .event(json!({"hook_event_name": "Stop", "session_id": SESSION, "timestamp": "2026-10-09T10:00:00Z",
                "cwd": "/home/dev/shop", "transcript_path": "/home/dev/.copilot/session-state/s/events.jsonl",
                "stop_reason": "end_turn", "stop_hook_active": false}))
            .expect("event");

        assert_eq!(event.moment, Moment::TurnEnd { continued: false });
        assert_eq!(event.name, "Stop");
        assert_eq!(
            event.transcript,
            Some(PathBuf::from(
                "/home/dev/.copilot/session-state/s/events.jsonl"
            ))
        );
        assert_eq!(
            scratch.moment(json!({"hook_event_name": "UserPromptSubmit", "prompt": "Fix paging"})),
            Some(Moment::Prompt("Fix paging".to_owned()))
        );
        assert_eq!(
            scratch.moment(json!({"hook_event_name": "Notification", "message": "x"})),
            None
        );
    }

    #[test]
    fn shell_failures_come_from_either_tool_event() {
        let scratch = Scratch::new("failures");
        let failing = "Error: Cannot find module 'left-pad'\n<exited with exit code 1>";

        assert_eq!(
            scratch.moment(
                json!({"toolName": "bash", "toolArgs": "{\"command\":\"npm test\"}",
                "toolResult": {"resultType": "success", "textResultForLlm": failing}})
            ),
            Some(Moment::CommandFailed(failing.to_owned()))
        );
        assert_eq!(
            scratch.moment(json!({"toolName": "bash", "toolArgs": {"command": "npm test"},
                "toolResult": {"resultType": "success", "textResultForLlm": "1 passing\n<exited with exit code 0>"}})),
            None
        );
        assert_eq!(
            scratch.moment(json!({"toolName": "view", "toolArgs": {"path": "a"},
                "toolResult": {"resultType": "success", "textResultForLlm": failing}})),
            None
        );
        assert_eq!(
            scratch.moment(json!({"toolName": "bash", "toolArgs": {"command": "npm test"}, "error": "Command timed out: npm test"})),
            Some(Moment::CommandFailed("Command timed out: npm test".to_owned()))
        );
        assert_eq!(
            scratch.moment(json!({"toolName": "bash", "toolArgs": {"command": "rm -rf /"}, "error": "The user rejected this tool call"})),
            None
        );
        assert_eq!(
            scratch.moment(json!({"hook_event_name": "PostToolUse", "tool_name": "bash", "tool_input": {"command": "npm test"},
                "tool_result": {"result_type": "success", "text_result_for_llm": failing}})),
            Some(Moment::CommandFailed(failing.to_owned()))
        );
    }

    #[test]
    fn the_transcript_is_found_by_session_id() {
        let scratch = Scratch::new("transcript");
        let events = scratch
            .dir
            .join("session-state")
            .join(SESSION)
            .join("events.jsonl");
        fs::create_dir_all(events.parent().expect("has a parent")).expect("dir is writable");
        fs::write(&events, "{}\n").expect("written");

        let event = scratch
            .event(json!({"sessionId": SESSION, "cwd": "/home/dev/shop", "reason": "complete"}))
            .expect("event");
        let traversal = scratch
            .event(json!({"sessionId": "../../etc", "cwd": "/home/dev/shop", "reason": "complete"}))
            .expect("event");

        assert_eq!(event.transcript, Some(events));
        assert_eq!(traversal.transcript, None);
        assert_eq!(
            Copilot::transcripts_in(&scratch.dir.join("session-state")).expect("sessions list"),
            [event.transcript.expect("found")]
        );
    }

    #[test]
    fn replies_are_json_objects() {
        let event = |moment| HookEvent {
            name: String::new(),
            session: "s".to_owned(),
            cwd: PathBuf::from("/w"),
            transcript: None,
            moment,
            observed: Vec::new(),
        };

        assert_eq!(
            Copilot.render(
                &event(Moment::Prompt("p".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some(r#"{"additionalContext":"<m/>"}"#.to_owned())
        );
        assert_eq!(
            Copilot.render(
                &event(Moment::CommandFailed("e".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some(r#"{"additionalContext":"<m/>"}"#.to_owned())
        );
        assert_eq!(
            Copilot.render(
                &event(Moment::TurnEnd { continued: false }),
                Reply::Remind("run it")
            ),
            Some(r#"{"decision":"block","reason":"run it"}"#.to_owned())
        );
        assert_eq!(
            Copilot.render(&event(Moment::SessionEnd), Reply::Remind("x")),
            None
        );
    }

    #[test]
    fn connect_owns_one_hooks_file() {
        let scratch = Scratch::new("connect");
        let mine = scratch.dir.join("hooks").join("mine.json");
        fs::create_dir_all(mine.parent().expect("has a parent")).expect("dir is writable");
        fs::write(
            &mine,
            r#"{"version":1,"hooks":{"sessionEnd":[{"bash":"afplay done.aiff"}]}}"#,
        )
        .expect("written");
        let program = Program::at("/home/dev/.cargo/bin/trodden");

        assert!(!Copilot::connected_at(&scratch.dir).expect("hooks load"));
        Scratch::apply(Copilot::connect_at(&scratch.dir, &program).expect("connect"));
        let hooks: Value = serde_json::from_str(
            &fs::read_to_string(Copilot::hooks(&scratch.dir)).expect("hooks are readable"),
        )
        .expect("hooks are JSON");

        assert!(Copilot::connected_at(&scratch.dir).expect("hooks load"));
        assert_eq!(hooks["version"], 1);
        assert_eq!(
            hooks["hooks"]["postToolUse"],
            json!([{"type": "command", "bash": "/home/dev/.cargo/bin/trodden hook copilot",
                "powershell": "& /home/dev/.cargo/bin/trodden hook copilot", "timeoutSec": 5, "matcher": "bash|powershell"}])
        );
        assert_eq!(hooks["hooks"].as_object().expect("events").len(), 7);
        assert!(
            Copilot::connect_at(&scratch.dir, &program)
                .expect("connect")
                .is_empty()
        );

        Scratch::apply(Copilot::disconnect_at(&scratch.dir).expect("disconnect"));
        assert!(!Copilot::connected_at(&scratch.dir).expect("hooks load"));
        assert!(!Copilot::hooks(&scratch.dir).exists());
        assert!(mine.exists(), "the user's own hooks stay");
        assert!(
            Copilot::disconnect_at(&scratch.dir)
                .expect("disconnect")
                .is_empty()
        );
    }
}
