use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use trodden_capture::droid::Transcript;
use trodden_core::Trace;
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply, claude_code::ClaudeHookInput};
use crate::connect::{HookFile, Program};

#[derive(Debug)]
pub(crate) struct Droid;

impl Droid {
    const HARNESS: &str = "droid";

    const SHELL: &str = "Execute";

    const TIMEOUT_SECONDS: u64 = 5;

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const MANUAL_STEP: &str = "Restart any running droid sessions to load the hooks.";

    const EVENTS: &[(&str, Option<&str>)] = &[
        ("SessionStart", None),
        ("UserPromptSubmit", None),
        ("PostToolUse", Some(Self::SHELL)),
        ("Stop", None),
        ("PreCompact", None),
        ("SessionEnd", None),
    ];

    fn config() -> Result<PathBuf> {
        let home = match env::var_os("FACTORY_HOME_OVERRIDE").filter(|home| !home.is_empty()) {
            Some(home) => PathBuf::from(home),
            None => env::home_dir().context("find the home directory")?,
        };
        Ok(home.join(".factory"))
    }

    fn hooks(config: &Path) -> PathBuf {
        config.join("hooks.json")
    }

    fn connect_at(config: &Path, program: &Program) -> Result<Vec<Change>> {
        let mut file = HookFile::load(Self::hooks(config))?;
        file.remove(&[], Self::HARNESS);
        let command = program.hook(Self::HARNESS);
        for (event, matcher) in Self::EVENTS {
            let mut group = json!({"hooks": [{
                "type": "command",
                "command": command,
                "timeout": Self::TIMEOUT_SECONDS,
            }]});
            if let Some(matcher) = matcher {
                group["matcher"] = json!(matcher);
            }
            file.add(&[], event, group)?;
        }
        Ok(file.change()?.into_iter().collect())
    }

    fn disconnect_at(config: &Path) -> Result<Vec<Change>> {
        let mut file = HookFile::load(Self::hooks(config))?;
        file.remove(&[], Self::HARNESS);
        Ok(file.change()?.into_iter().collect())
    }

    fn connected_at(config: &Path) -> Result<bool> {
        Ok(HookFile::load(Self::hooks(config))?.contains(&[], Self::HARNESS))
    }

    fn transcripts_in(sessions: &Path) -> Result<Vec<PathBuf>> {
        let mut transcripts = Vec::new();
        let mut directories = vec![sessions.to_path_buf()];
        for entry in
            fs::read_dir(sessions).with_context(|| format!("list {}", sessions.display()))?
        {
            let path = entry.context("read a session directory entry")?.path();
            if path.is_dir() {
                directories.push(path);
            }
        }
        for directory in directories {
            let Ok(entries) = fs::read_dir(&directory) else {
                continue;
            };
            for path in entries.flatten().map(|entry| entry.path()) {
                if path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
                    && !Self::is_subagent(&path.with_extension("settings.json"))
                {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }

    fn is_subagent(settings: &Path) -> bool {
        Self::settings(settings).is_some_and(|settings| {
            settings
                .get("tags")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|tag| tag.pointer("/metadata/callingSessionId").is_some())
        })
    }

    fn settings(path: &Path) -> Option<Value> {
        serde_json::from_str(&fs::read_to_string(path).ok()?).ok()
    }

    fn model(sessions: &Path, text: &str) -> Option<String> {
        let session = Transcript::session(text)?;
        let name = format!("{session}.settings.json");
        let slug =
            Transcript::working_directory(text).map(|cwd| cwd.replace(['/', '\\', ':'], "-"));
        let mut candidates: Vec<PathBuf> = slug
            .map(|slug| sessions.join(slug).join(&name))
            .into_iter()
            .collect();
        candidates.push(sessions.join(&name));
        if let Ok(entries) = fs::read_dir(sessions) {
            candidates.extend(entries.flatten().map(|entry| entry.path().join(&name)));
        }
        candidates.into_iter().find_map(|path| {
            Self::settings(&path)?
                .get("model")
                .and_then(Value::as_str)
                .filter(|model| !model.is_empty())
                .map(str::to_owned)
        })
    }

    fn parse_in(
        sessions: Option<&Path>,
        text: &str,
        redactor: &Redactor,
    ) -> Result<(Trace, Option<PathBuf>)> {
        let model = sessions.and_then(|sessions| Self::model(sessions, text));
        Ok((
            Transcript::parse(text, model.as_deref(), redactor)?,
            Transcript::working_directory(text).map(PathBuf::from),
        ))
    }

    fn failure(response: Option<&Value>) -> Option<String> {
        match response? {
            Value::String(text) => {
                let code = Self::exit_code(text)?;
                (code != 0).then(|| text.clone())
            }
            Value::Object(fields) => {
                let flag = |key: &str| fields.get(key).and_then(Value::as_bool);
                if flag("interrupted") == Some(true) || flag("isInterrupt") == Some(true) {
                    return None;
                }
                let code = [
                    "exitCode",
                    "exit_code",
                    "returnCode",
                    "return_code",
                    "code",
                    "status",
                ]
                .iter()
                .find_map(|key| fields.get(*key).and_then(Value::as_i64));
                let error = fields
                    .get("error")
                    .and_then(Value::as_str)
                    .filter(|error| !error.is_empty());
                let output: Vec<&str> = ["stdout", "stderr", "output", "content", "result"]
                    .iter()
                    .filter_map(|key| fields.get(*key).and_then(Value::as_str))
                    .chain(error)
                    .filter(|text| !text.trim().is_empty())
                    .collect();
                let output = output.join("\n");
                let failed = match code {
                    Some(code) => code != 0,
                    None => {
                        flag("is_error") == Some(true)
                            || flag("isError") == Some(true)
                            || flag("success") == Some(false)
                            || error.is_some()
                            || Self::exit_code(&output).is_some_and(|code| code != 0)
                    }
                };
                failed.then_some(output)
            }
            _ => None,
        }
    }

    fn exit_code(text: &str) -> Option<i64> {
        let lines: Vec<&str> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        [lines.first(), lines.last()]
            .into_iter()
            .flatten()
            .find_map(|line| {
                let lower = line.to_ascii_lowercase();
                ["exit code:", "exit code", "exited with code", "exit status"]
                    .iter()
                    .find_map(|marker| {
                        let rest = &lower[lower.find(marker)? + marker.len()..];
                        let digits: String = rest
                            .trim_start_matches([' ', ':', '='])
                            .chars()
                            .take_while(|c| c.is_ascii_digit() || *c == '-')
                            .collect();
                        digits.parse().ok()
                    })
            })
    }
}

impl Agent for Droid {
    fn title(&self) -> &'static str {
        "Factory Droid"
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(Some(Self::config()?.join("sessions")))
    }

    fn transcripts(&self, sessions: &Path) -> Result<Vec<PathBuf>> {
        Self::transcripts_in(sessions)
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        let sessions = Self::config().ok().map(|config| config.join("sessions"));
        Self::parse_in(sessions.as_deref(), text, redactor)
    }

    fn event(&self, payload: &str, _redactor: &Redactor) -> Result<Option<HookEvent>> {
        let Some(input) = ClaudeHookInput::parse(payload)? else {
            return Ok(None);
        };
        let moment = match input.hook_event_name.as_str() {
            "SessionStart" => Moment::SessionStart,
            "UserPromptSubmit" => match &input.prompt {
                Some(prompt) => Moment::Prompt(prompt.clone()),
                None => return Ok(None),
            },
            "PostToolUse" if input.tool_name.as_deref() == Some(Self::SHELL) => {
                match Self::failure(input.tool_response.as_ref()) {
                    Some(error) => Moment::CommandFailed(error),
                    None => return Ok(None),
                }
            }
            "PostToolUseFailure" => match input.failure(&[Self::SHELL]) {
                Some(error) => Moment::CommandFailed(error),
                None => return Ok(None),
            },
            "Stop" => Moment::TurnEnd {
                continued: input.stop_hook_active,
            },
            "PreCompact" => Moment::Compacting,
            "SessionEnd" => Moment::SessionEnd,
            _ => return Ok(None),
        };
        Ok(input.event(moment))
    }

    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_), Reply::Recall(envelope)) => Some(envelope.to_owned()),
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
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-droid-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn write(&self, path: &str, contents: &str) -> PathBuf {
            let path = self.dir.join(path);
            fs::create_dir_all(path.parent().expect("has a parent")).expect("dir is writable");
            fs::write(&path, contents).expect("file is writable");
            path
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }

        fn hooks(&self) -> Value {
            serde_json::from_str(
                &fs::read_to_string(self.dir.join("hooks.json")).expect("readable"),
            )
            .expect("JSON")
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
        fn moment(extra: Value) -> Option<Moment> {
            let mut payload = json!({"session_id": "6f0c", "cwd": "/home/dev/shop",
                                     "transcript_path": "/home/dev/.factory/sessions/-home-dev-shop/6f0c.jsonl",
                                     "permission_mode": "auto-low"});
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            Droid
                .event(&payload.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
                .map(|event| event.moment)
        }

        fn event(moment: Moment) -> HookEvent {
            HookEvent {
                name: String::new(),
                session: "6f0c".to_owned(),
                cwd: PathBuf::from("/home/dev/shop"),
                transcript: None,
                moment,
                observed: Vec::new(),
            }
        }
    }

    #[test]
    fn hook_events_map_to_moments() {
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "UserPromptSubmit", "prompt": "Fix paging", "has_images": false})
            ),
            Some(Moment::Prompt("Fix paging".to_owned()))
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "Stop", "stop_hook_active": true, "tool_execution_count": 3})
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
            Payload::moment(json!({"hook_event_name": "SessionStart", "source": "startup"})),
            Some(Moment::SessionStart)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "Notification"})),
            None
        );
    }

    #[test]
    fn failed_executes_are_read_from_any_plausible_response() {
        let post = |tool: &str, response: Value| {
            Payload::moment(json!({"hook_event_name": "PostToolUse", "tool_name": tool,
                                   "tool_input": {"command": "npm test"}, "tool_response": response}))
        };
        let failed = |text: &str| Some(Moment::CommandFailed(text.to_owned()));

        assert_eq!(
            post(
                "Execute",
                json!({"exitCode": 1, "stdout": "", "stderr": "Error: Cannot find module 'x'"})
            ),
            failed("Error: Cannot find module 'x'")
        );
        assert_eq!(
            post("Execute", json!({"exit_code": 0, "stdout": "1 passing"})),
            None
        );
        assert_eq!(
            post("Execute", json!({"success": false, "output": "boom"})),
            failed("boom")
        );
        assert_eq!(
            post(
                "Execute",
                json!({"is_error": true, "content": "Exit code 2\nboom"})
            ),
            failed("Exit code 2\nboom")
        );
        assert_eq!(
            post("Execute", json!("error: tests failed\nExit code: 101")),
            failed("error: tests failed\nExit code: 101")
        );
        assert_eq!(post("Execute", json!("all good")), None);
        assert_eq!(
            post("Execute", json!({"exitCode": 130, "interrupted": true})),
            None
        );
        assert_eq!(post("Edit", json!({"exitCode": 1, "stderr": "x"})), None);
        assert_eq!(post("Execute", Value::Null), None);
    }

    #[test]
    fn replies_take_the_shape_droid_reads() {
        assert_eq!(
            Droid.render(
                &Payload::event(Moment::Prompt("p".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some("<m/>".to_owned())
        );
        let failure: Value = serde_json::from_str(
            &Droid
                .render(
                    &Payload::event(Moment::CommandFailed("e".to_owned())),
                    Reply::Recall("<m/>"),
                )
                .expect("failure recall renders"),
        )
        .expect("JSON");
        assert_eq!(
            failure["hookSpecificOutput"]["hookEventName"],
            "PostToolUse"
        );
        assert_eq!(failure["hookSpecificOutput"]["additionalContext"], "<m/>");
        assert_eq!(
            Droid.render(
                &Payload::event(Moment::TurnEnd { continued: false }),
                Reply::Remind("run it")
            ),
            Some(r#"{"decision":"block","reason":"run it"}"#.to_owned())
        );
    }

    #[test]
    fn connect_keeps_other_hooks_and_round_trips() {
        let scratch = Scratch::new("connect");
        scratch.write(
            "hooks.json",
            &json!({"PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "prettier --write"}]}]})
                .to_string(),
        );
        let program = Program::at("/home/dev/.cargo/bin/trodden");

        assert!(!Droid::connected_at(&scratch.dir).expect("readable"));
        Scratch::apply(Droid::connect_at(&scratch.dir, &program).expect("connects"));
        assert!(
            Droid::connect_at(&scratch.dir, &program)
                .expect("connects")
                .is_empty(),
            "idempotent"
        );
        assert!(Droid::connected_at(&scratch.dir).expect("readable"));

        let hooks = scratch.hooks();
        assert_eq!(hooks["PostToolUse"][0]["matcher"], "Edit");
        assert_eq!(hooks["PostToolUse"][1]["matcher"], "Execute");
        assert_eq!(
            hooks["Stop"][0]["hooks"][0],
            json!({"type": "command", "command": "/home/dev/.cargo/bin/trodden hook droid", "timeout": 5})
        );
        assert!(
            hooks.get("hooks").is_none(),
            "user hooks are a bare event map"
        );

        Scratch::apply(Droid::disconnect_at(&scratch.dir).expect("disconnects"));
        assert_eq!(
            scratch.hooks(),
            json!({"PostToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "prettier --write"}]}]})
        );
        assert!(!Droid::connected_at(&scratch.dir).expect("readable"));
        assert!(Droid::MANUAL_STEP.contains("Restart"));
    }

    #[test]
    fn backfill_finds_main_sessions_and_their_model() {
        let scratch = Scratch::new("history");
        let header = json!({"type": "session_start", "id": "6f0c", "cwd": "/home/dev/shop"});
        let prompt = json!({"type": "message", "id": "m1", "timestamp": "2026-10-09T10:00:00Z",
                            "message": {"role": "user", "content": "Fix the paging bug"}});
        let text = format!("{header}\n{prompt}\n");
        let main = scratch.write("-home-dev-shop/6f0c.jsonl", &text);
        scratch.write(
            "-home-dev-shop/6f0c.settings.json",
            r#"{"model": "claude-opus-4-5"}"#,
        );
        scratch.write("-home-dev-shop/sub1.jsonl", &text);
        scratch.write(
            "-home-dev-shop/sub1.settings.json",
            r#"{"tags": [{"name": "subagent", "metadata": {"callingSessionId": "6f0c", "callingToolUseId": "t1"}}]}"#,
        );
        let legacy = scratch.write("legacy.jsonl", &text);

        assert_eq!(
            Droid::transcripts_in(&scratch.dir).expect("listed"),
            [main, legacy]
        );
        let (trace, cwd) =
            Droid::parse_in(Some(&scratch.dir), &text, &Redactor::with_home("/home/dev"))
                .expect("parses");
        assert_eq!(trace.model.as_deref(), Some("claude-opus-4-5"));
        assert_eq!(cwd, Some(PathBuf::from("/home/dev/shop")));
        let (trace, _) =
            Droid::parse_in(None, &text, &Redactor::with_home("/home/dev")).expect("parses");
        assert_eq!(trace.model, None);
    }
}
