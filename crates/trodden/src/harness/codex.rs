use std::{
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use trodden_capture::codex::Rollout;
use trodden_core::Trace;
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply, claude_code::ClaudeHookInput};
use crate::connect::{HookFile, Program};

#[derive(Debug, Deserialize)]
struct CodexHookInput {
    session_id: String,
    #[serde(default)]
    transcript_path: Option<PathBuf>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    hook_event_name: String,
    #[serde(default)]
    agent_id: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_input: Option<Value>,
    #[serde(default)]
    tool_response: Option<Value>,
    #[serde(default)]
    stop_hook_active: bool,
}

impl CodexHookInput {
    fn failure(&self, redactor: &Redactor) -> Option<String> {
        if self.tool_name.as_deref() != Some(Codex::SHELL) {
            return None;
        }
        let command = self
            .tool_input
            .as_ref()
            .and_then(|input| input.get("command"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let output = match self.tool_response.as_ref()? {
            Value::String(text) => text.clone(),
            Value::Object(fields) => ["output", "stdout", "stderr", "aggregated_output"]
                .iter()
                .filter_map(|key| fields.get(*key).and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => return None,
        };
        Rollout::failed(command, &output, redactor).then_some(output)
    }
}

#[derive(Debug)]
pub(crate) struct Codex;

impl Codex {
    const HARNESS: &str = "codex";

    const SHELL: &str = "Bash";

    const TIMEOUT_SECONDS: u64 = 5;

    const SESSION_END_TIMEOUT_SECONDS: u64 = 3;

    // Trodden never writes Codex's trust hashes: that would skip the user's review.
    pub(crate) const MANUAL_STEP: &str = "Start codex and choose \"Trust all and continue\" when it asks you to review the new hooks (or trust them in /hooks).";

    const EVENTS: &[(&str, Option<&str>)] = &[
        ("SessionStart", None),
        ("UserPromptSubmit", None),
        ("PostToolUse", Some("^Bash$")),
        ("Stop", None),
        ("PreCompact", None),
        ("SessionEnd", None),
    ];

    fn config() -> Result<PathBuf> {
        match env::var_os("CODEX_HOME").filter(|home| !home.is_empty()) {
            Some(home) => Ok(PathBuf::from(home)),
            None => Ok(env::home_dir()
                .context("find the home directory")?
                .join(".codex")),
        }
    }

    fn hooks(config: &Path) -> PathBuf {
        config.join("hooks.json")
    }

    // Codex runs hooks through `$SHELL -lc`, so the command stays a bare path and arguments
    // that fish reads too.
    fn connect_at(config: &Path, program: &Program) -> Result<Vec<Change>> {
        let mut file = HookFile::load(Self::hooks(config))?;
        file.remove(&["hooks"], Self::HARNESS);
        let command = program.hook(Self::HARNESS);
        for (event, matcher) in Self::EVENTS {
            let timeout = if *event == "SessionEnd" {
                Self::SESSION_END_TIMEOUT_SECONDS
            } else {
                Self::TIMEOUT_SECONDS
            };
            let mut group = json!({"hooks": [{
                "type": "command",
                "command": command,
                "timeout": timeout,
            }]});
            if let Some(matcher) = matcher {
                group["matcher"] = json!(matcher);
            }
            file.add(&["hooks"], event, group)?;
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

    fn transcripts_in(sessions: &Path) -> Result<Vec<PathBuf>> {
        let mut directories = vec![sessions.to_path_buf()];
        if sessions.file_name().is_some_and(|name| name == "sessions")
            && let Some(archived) = sessions
                .parent()
                .map(|codex| codex.join("archived_sessions"))
            && archived.is_dir()
        {
            directories.push(archived);
        }
        fs::read_dir(sessions).with_context(|| format!("list {}", sessions.display()))?;
        let mut transcripts = Vec::new();
        while let Some(directory) = directories.pop() {
            let Ok(entries) = fs::read_dir(&directory) else {
                continue;
            };
            for path in entries.flatten().map(|entry| entry.path()) {
                if path.is_dir() {
                    directories.push(path);
                } else if Self::is_rollout(&path) && !Self::is_subagent(&path) {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }

    fn is_rollout(path: &Path) -> bool {
        path.extension()
            .is_some_and(|extension| extension == "jsonl")
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("rollout-"))
    }

    fn is_subagent(path: &Path) -> bool {
        let Ok(file) = fs::File::open(path) else {
            return false;
        };
        let mut first = String::new();
        BufReader::new(file).read_line(&mut first).is_ok() && Rollout::is_subagent(&first)
    }
}

impl Agent for Codex {
    fn title(&self) -> &'static str {
        "Codex"
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(Some(Self::config()?.join("sessions")))
    }

    fn transcripts(&self, history: &Path) -> Result<Vec<PathBuf>> {
        Self::transcripts_in(history)
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        Ok((
            Rollout::parse(text, redactor)?,
            Rollout::working_directory(text),
        ))
    }

    fn event(&self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>> {
        let input: CodexHookInput =
            serde_json::from_str(payload).context("parse the hook payload")?;
        if input.agent_id.is_some() {
            return Ok(None);
        }
        let moment = match input.hook_event_name.as_str() {
            "SessionStart" => Moment::SessionStart,
            "UserPromptSubmit" => match &input.prompt {
                Some(prompt) => Moment::Prompt(prompt.clone()),
                None => return Ok(None),
            },
            "PostToolUse" => match input.failure(redactor) {
                Some(output) => Moment::CommandFailed(output),
                None => return Ok(None),
            },
            "Stop" => Moment::TurnEnd {
                continued: input.stop_hook_active,
            },
            "PreCompact" => Moment::Compacting,
            "SessionEnd" => Moment::SessionEnd,
            _ => return Ok(None),
        };
        let Some(cwd) = input.cwd.filter(|cwd| cwd.is_absolute()) else {
            return Ok(None);
        };
        Ok(Some(HookEvent {
            name: input.hook_event_name,
            session: input.session_id,
            cwd,
            transcript: input
                .transcript_path
                .filter(|path| !path.as_os_str().is_empty()),
            moment,
            observed: Vec::new(),
        }))
    }

    // Codex drops a reply with unknown fields or another event's `hookEventName`.
    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_), Reply::Recall(envelope)) => Some(ClaudeHookInput::recall_context(
                "UserPromptSubmit",
                envelope,
            )),
            (Moment::CommandFailed(_), Reply::Recall(envelope)) => {
                Some(ClaudeHookInput::recall_context("PostToolUse", envelope))
            }
            (Moment::TurnEnd { .. }, Reply::Remind(reason)) => Some(ClaudeHookInput::block(reason)),
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

    #[derive(Debug)]
    struct Payload;

    impl Payload {
        fn event(extra: Value) -> Option<HookEvent> {
            let mut payload = json!({
                "session_id": "019f0a10-5c2e-7d41-9b8a-3e6f1c2d4a5b",
                "turn_id": "019f0a13-0000-7000-8000-000000000003",
                "transcript_path": "/home/dev/.codex/sessions/2026/06/07/rollout-2026-06-07T17-48-19-019f0a10-5c2e-7d41-9b8a-3e6f1c2d4a5b.jsonl",
                "cwd": "/home/dev/shop",
                "model": "gpt-5.5",
                "permission_mode": "default",
            });
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            Codex
                .event(&payload.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
        }

        fn moment(extra: Value) -> Option<Moment> {
            Self::event(extra).map(|event| event.moment)
        }

        fn rendered(moment: Moment, reply: Reply<'_>) -> Option<String> {
            let event = HookEvent {
                name: String::new(),
                session: "s".to_owned(),
                cwd: PathBuf::from("/home/dev/shop"),
                transcript: None,
                moment,
                observed: Vec::new(),
            };
            Codex.render(&event, reply)
        }
    }

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-codex-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }

        fn hooks(&self) -> Value {
            serde_json::from_str(
                &fs::read_to_string(Codex::hooks(&self.dir)).expect("hooks are readable"),
            )
            .expect("hooks are JSON")
        }

        fn write(&self, relative: &str, text: &str) -> PathBuf {
            let path = self.dir.join(relative);
            fs::create_dir_all(path.parent().expect("has a parent")).expect("dir is writable");
            fs::write(&path, text).expect("file is writable");
            path
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn hook_events_map_to_moments() {
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "UserPromptSubmit", "prompt": "Fix paging"})),
            Some(Moment::Prompt("Fix paging".to_owned()))
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "SessionStart", "source": "startup"})),
            Some(Moment::SessionStart)
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "Stop", "stop_hook_active": true, "last_assistant_message": null})
            ),
            Some(Moment::TurnEnd { continued: true })
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "PreCompact", "trigger": "auto"})),
            Some(Moment::Compacting)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "SessionEnd", "reason": "other"})),
            Some(Moment::SessionEnd)
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "SubagentStop", "agent_id": "a", "agent_type": "worker"})
            ),
            None
        );
        let event = Payload::event(json!({"hook_event_name": "Stop"})).expect("event");
        assert_eq!(event.session, "019f0a10-5c2e-7d41-9b8a-3e6f1c2d4a5b");
        assert!(event.transcript.is_some_and(|path| {
            path.ends_with("rollout-2026-06-07T17-48-19-019f0a10-5c2e-7d41-9b8a-3e6f1c2d4a5b.jsonl")
        }));
    }

    #[test]
    fn subagent_events_are_left_to_the_parent_session() {
        let failing =
            "error[E0425]: cannot find value `total` in this scope\n --> src/lib.rs:3:5\n";
        let subagent = json!({"session_id": "019f0c00-0000-7000-8000-00000000c41d",
            "agent_id": "019f0c00-0000-7000-8000-00000000c41d", "agent_type": "worker"});
        let events = [
            json!({"hook_event_name": "UserPromptSubmit", "prompt": "Review the diff"}),
            json!({"hook_event_name": "PostToolUse", "tool_name": "Bash",
                "tool_input": {"command": "cargo test"}, "tool_response": failing}),
            json!({"hook_event_name": "Stop", "stop_hook_active": false}),
            json!({"hook_event_name": "PreCompact", "trigger": "auto"}),
            json!({"hook_event_name": "SessionStart", "source": "startup"}),
            json!({"hook_event_name": "SessionEnd", "reason": "other"}),
        ];

        for mut event in events {
            assert!(Payload::moment(event.clone()).is_some(), "{event}");
            event
                .as_object_mut()
                .expect("object")
                .extend(subagent.as_object().expect("object").clone());
            assert_eq!(Payload::moment(event.clone()), None, "{event}");
        }
    }

    #[test]
    fn odd_payloads_are_ignored() {
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "Stop", "cwd": "shop"})),
            None
        );
        let event = Payload::event(json!({"hook_event_name": "Stop", "transcript_path": null}))
            .expect("event");
        assert_eq!(event.transcript, None);
        assert!(Codex.event("not json", &Redactor::new()).is_err());
    }

    #[test]
    fn failed_shell_commands_are_read_from_their_output() {
        let failing = "   Compiling shop v0.1.0\nerror[E0425]: cannot find value `total` in this scope\n --> src/lib.rs:3:5\n";

        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUse", "tool_name": "Bash",
                "tool_input": {"command": "cargo test"}, "tool_response": failing, "tool_use_id": "call_1"})
            ),
            Some(Moment::CommandFailed(failing.to_owned()))
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUse", "tool_name": "Bash",
                "tool_input": {"command": "cargo test"}, "tool_response": "test result: ok. 3 passed; 0 failed\n"})
            ),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUse", "tool_name": "Bash",
                "tool_input": {"command": "cat src/lib.rs"}, "tool_response": failing})
            ),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUse", "tool_name": "apply_patch",
                "tool_input": {"command": "*** Begin Patch"}, "tool_response": failing})
            ),
            None
        );
    }

    #[test]
    fn replies_take_the_shape_codex_accepts() {
        let prompt = Payload::rendered(Moment::Prompt("p".to_owned()), Reply::Recall("<m/>"))
            .expect("prompt recall renders");
        assert_eq!(
            serde_json::from_str::<Value>(&prompt).expect("JSON"),
            json!({"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": "<m/>"}})
        );
        let failure =
            Payload::rendered(Moment::CommandFailed("e".to_owned()), Reply::Recall("<m/>"))
                .expect("failure recall renders");
        assert_eq!(
            serde_json::from_str::<Value>(&failure).expect("JSON"),
            json!({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": "<m/>"}})
        );
        assert_eq!(
            Payload::rendered(
                Moment::TurnEnd { continued: false },
                Reply::Remind("run it")
            ),
            Some(r#"{"decision":"block","reason":"run it"}"#.to_owned())
        );
        assert_eq!(
            Payload::rendered(Moment::Compacting, Reply::Recall("<m/>")),
            None
        );
    }

    #[test]
    fn connect_keeps_the_users_hooks_and_round_trips() {
        let scratch = Scratch::new("connect");
        scratch.write(
            "hooks.json",
            r#"{"hooks":{"SessionStart":[{"hooks":[{"command":"bash '/home/dev/.codex/notify.sh' session","timeout":10,"type":"command"}]}]}}"#,
        );
        let program = Program::at("/home/dev/.cargo/bin/trodden");

        assert!(!Codex::connected_at(&scratch.dir).expect("hooks load"));
        Scratch::apply(Codex::connect_at(&scratch.dir, &program).expect("connect"));
        let hooks = scratch.hooks();

        assert!(Codex::connected_at(&scratch.dir).expect("hooks load"));
        assert_eq!(
            hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"],
            "bash '/home/dev/.codex/notify.sh' session"
        );
        assert_eq!(
            hooks["hooks"]["SessionStart"][1]["hooks"][0]["command"],
            "/home/dev/.cargo/bin/trodden hook codex"
        );
        assert_eq!(
            hooks["hooks"]["PostToolUse"],
            json!([{"matcher": "^Bash$", "hooks": [{"type": "command", "command": "/home/dev/.cargo/bin/trodden hook codex", "timeout": 5}]}])
        );
        assert_eq!(hooks["hooks"]["SessionEnd"][0]["hooks"][0]["timeout"], 3);
        let events: Vec<&String> = hooks["hooks"].as_object().expect("events").keys().collect();
        assert_eq!(
            events,
            [
                "SessionStart",
                "UserPromptSubmit",
                "PostToolUse",
                "Stop",
                "PreCompact",
                "SessionEnd"
            ]
        );

        assert!(
            Codex::connect_at(&scratch.dir, &program)
                .expect("connect again")
                .is_empty(),
            "connecting twice changes nothing"
        );

        Scratch::apply(Codex::disconnect_at(&scratch.dir).expect("disconnect"));
        assert!(!Codex::connected_at(&scratch.dir).expect("hooks load"));
        assert_eq!(
            scratch.hooks(),
            json!({"hooks":{"SessionStart":[{"hooks":[{"command":"bash '/home/dev/.codex/notify.sh' session","timeout":10,"type":"command"}]}]}})
        );
        assert!(!Codex::MANUAL_STEP.is_empty());
    }

    #[test]
    fn backfill_finds_rollouts_but_not_subagents() {
        let scratch = Scratch::new("backfill");
        let meta = |source: Value| {
            format!(
                "{}\n",
                json!({"timestamp": "2026-06-07T12:21:17.584Z", "type": "session_meta",
                       "payload": {"id": "019f0a10", "cwd": "/home/dev/shop", "source": source}})
            )
        };
        let main = scratch.write(
            "sessions/2026/06/07/rollout-2026-06-07T17-48-19-019f0a10.jsonl",
            &meta(json!("cli")),
        );
        scratch.write(
            "sessions/2026/06/07/rollout-2026-06-07T17-50-00-019f0c00.jsonl",
            &meta(
                json!({"subagent": {"thread_spawn": {"parent_thread_id": "019f0a10", "depth": 1}}}),
            ),
        );
        scratch.write("sessions/2026/06/07/notes.txt", "not a rollout");
        let archived = scratch.write(
            "archived_sessions/rollout-2026-05-01T09-00-00-019e0000.jsonl",
            &meta(json!("exec")),
        );

        let transcripts =
            Codex::transcripts_in(&scratch.dir.join("sessions")).expect("sessions list");

        assert_eq!(transcripts, [archived, main]);
    }
}
