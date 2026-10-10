use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use trodden_capture::claude_code::Transcript;
use trodden_core::Trace;
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply};
use crate::connect::{HookFile, Program};

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct ClaudeHookInput {
    pub(crate) session_id: String,
    #[serde(default)]
    pub(crate) transcript_path: Option<PathBuf>,
    pub(crate) cwd: Option<PathBuf>,
    pub(crate) hook_event_name: String,
    #[serde(default)]
    pub(crate) prompt: Option<String>,
    #[serde(default)]
    pub(crate) tool_name: Option<String>,
    #[serde(default)]
    pub(crate) tool_response: Option<Value>,
    #[serde(default)]
    pub(crate) error: Option<String>,
    #[serde(default)]
    pub(crate) is_interrupt: bool,
    #[serde(default)]
    pub(crate) stop_hook_active: bool,
    #[serde(default)]
    pub(crate) cursor_version: Option<String>,
}

impl ClaudeHookInput {
    pub(crate) fn parse(payload: &str) -> Result<Option<Self>> {
        let input: Self = serde_json::from_str(payload).context("parse the hook payload")?;
        if input.cursor_version.is_some() || env::var_os("CURSOR_VERSION").is_some() {
            return Ok(None);
        }
        Ok(Some(input))
    }

    pub(crate) fn event(self, moment: Moment) -> Option<HookEvent> {
        let cwd = self.cwd.filter(|cwd| cwd.is_absolute())?;
        Some(HookEvent {
            name: self.hook_event_name,
            session: self.session_id,
            cwd,
            transcript: self
                .transcript_path
                .filter(|path| !path.as_os_str().is_empty()),
            moment,
            observed: Vec::new(),
        })
    }

    pub(crate) fn failure(&self, shell: &[&str]) -> Option<String> {
        let tool = self.tool_name.as_deref()?;
        if !shell.contains(&tool) || self.is_interrupt {
            return None;
        }
        self.error.clone()
    }

    pub(crate) fn recall_context(event: &str, envelope: &str) -> String {
        json!({
            "hookSpecificOutput": {
                "hookEventName": event,
                "additionalContext": envelope,
            }
        })
        .to_string()
    }

    pub(crate) fn block(reason: &str) -> String {
        json!({ "decision": "block", "reason": reason }).to_string()
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClaudeCodeLine {
    cwd: Option<PathBuf>,
    #[serde(default)]
    is_sidechain: bool,
}

#[derive(Debug)]
pub(crate) struct ClaudeCode;

impl ClaudeCode {
    fn working_directory(text: &str) -> Option<PathBuf> {
        text.lines()
            .filter_map(|line| serde_json::from_str::<ClaudeCodeLine>(line).ok())
            .filter(|line| !line.is_sidechain)
            .find_map(|line| line.cwd)
    }

    const HARNESS: &str = "claude-code";

    const TIMEOUT_SECONDS: u64 = 5;

    const EVENTS: &[(&str, Option<&str>)] = &[
        ("SessionStart", None),
        ("UserPromptSubmit", None),
        ("PostToolUseFailure", Some("Bash")),
        ("Stop", None),
        ("PreCompact", None),
        ("SessionEnd", None),
    ];

    fn config() -> Result<PathBuf> {
        match env::var_os("CLAUDE_CONFIG_DIR").filter(|dir| !dir.is_empty()) {
            Some(dir) => Ok(PathBuf::from(dir)),
            None => Ok(env::home_dir()
                .context("find the home directory")?
                .join(".claude")),
        }
    }

    fn settings(config: &Path) -> PathBuf {
        config.join("settings.json")
    }

    // The plugin runs the same hooks; both at once would recall twice per prompt.
    fn has_plugin(config: &Path) -> bool {
        fs::read_to_string(config.join("plugins").join("installed_plugins.json"))
            .is_ok_and(|text| text.contains("trodden@trodden"))
    }

    fn connect_at(config: &Path, program: &Program) -> Result<Vec<Change>> {
        if Self::has_plugin(config) {
            bail!(
                "the Trodden plugin for Claude Code is installed and already runs these hooks; \
                 keep it, or run `claude plugin uninstall trodden@trodden` and connect again"
            );
        }
        let mut file = HookFile::load(Self::settings(config))?;
        file.remove(&["hooks"], Self::HARNESS);
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
            file.add(&["hooks"], event, group)?;
        }
        Ok(file.change()?.into_iter().collect())
    }

    fn disconnect_at(config: &Path) -> Result<Vec<Change>> {
        let mut file = HookFile::load(Self::settings(config))?;
        let removed = file.remove(&["hooks"], Self::HARNESS);
        if !removed && Self::has_plugin(config) {
            bail!(
                "Trodden runs in Claude Code as a plugin; run `claude plugin uninstall trodden@trodden`"
            );
        }
        Ok(file.change()?.into_iter().collect())
    }

    fn connected_at(config: &Path) -> Result<bool> {
        Ok(Self::has_plugin(config)
            || HookFile::load(Self::settings(config))?.contains(&["hooks"], Self::HARNESS))
    }
}

impl Agent for ClaudeCode {
    fn title(&self) -> &'static str {
        "Claude Code"
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(Some(Self::config()?.join("projects")))
    }

    fn transcripts(&self, projects: &Path) -> Result<Vec<PathBuf>> {
        let mut transcripts = Vec::new();
        for project in
            fs::read_dir(projects).with_context(|| format!("list {}", projects.display()))?
        {
            let project = project.context("read a project directory entry")?.path();
            let Ok(entries) = fs::read_dir(&project) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
                {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        Ok((
            Transcript::parse(text, redactor)?,
            Self::working_directory(text),
        ))
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
            "PostToolUseFailure" => match input.failure(&["Bash"]) {
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
            (Moment::CommandFailed(_), Reply::Recall(envelope)) => Some(
                ClaudeHookInput::recall_context("PostToolUseFailure", envelope),
            ),
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
    use super::*;

    #[derive(Debug)]
    struct Payload;

    impl Payload {
        fn event(payload: Value) -> Option<HookEvent> {
            ClaudeCode
                .event(&payload.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
        }
    }

    #[test]
    fn invalid_bytes_only_spoil_their_own_line() {
        let bytes = [
            b"{\"type\":\"user\",\"note\":\"caf\xe9\"}\n".as_slice(),
            b"{\"cwd\":\"/home/dev/shop\"}\n",
            b"{\"cwd\":\"/home/dev/caf\xc3",
        ]
        .concat();
        let text = String::from_utf8_lossy(&bytes).into_owned();

        assert_eq!(
            ClaudeCode::working_directory(&text),
            Some(PathBuf::from("/home/dev/shop"))
        );
    }

    #[test]
    fn hook_events_map_to_moments() {
        let base =
            json!({"session_id": "s", "cwd": "/home/dev/shop", "transcript_path": "/t.jsonl"});
        let with = |extra: Value| {
            let mut payload = base.clone();
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            Payload::event(payload).map(|event| event.moment)
        };

        assert_eq!(
            with(json!({"hook_event_name": "UserPromptSubmit", "prompt": "Fix paging"})),
            Some(Moment::Prompt("Fix paging".to_owned()))
        );
        assert_eq!(
            with(
                json!({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "error": "boom"})
            ),
            Some(Moment::CommandFailed("boom".to_owned()))
        );
        assert_eq!(
            with(
                json!({"hook_event_name": "PostToolUseFailure", "tool_name": "Bash", "error": "x", "is_interrupt": true})
            ),
            None
        );
        assert_eq!(
            with(
                json!({"hook_event_name": "PostToolUseFailure", "tool_name": "Read", "error": "x"})
            ),
            None
        );
        assert_eq!(
            with(json!({"hook_event_name": "Stop", "stop_hook_active": true})),
            Some(Moment::TurnEnd { continued: true })
        );
        assert_eq!(
            with(json!({"hook_event_name": "SessionEnd"})),
            Some(Moment::SessionEnd)
        );
        assert_eq!(with(json!({"hook_event_name": "Notification"})), None);
    }

    #[test]
    fn foreign_payloads_are_ignored() {
        assert_eq!(
            Payload::event(
                json!({"session_id": "s", "hook_event_name": "Stop", "cwd": "/w", "cursor_version": "3.23"})
            ),
            None
        );
        assert_eq!(
            Payload::event(json!({"session_id": "s", "hook_event_name": "Stop"})),
            None
        );
        let empty = Payload::event(json!({"session_id": "s", "hook_event_name": "Stop", "cwd": "/w", "transcript_path": ""}))
            .expect("event");
        assert_eq!(empty.transcript, None);
    }

    #[test]
    fn replies_take_the_shape_each_event_needs() {
        let event = |moment| HookEvent {
            name: String::new(),
            session: "s".to_owned(),
            cwd: PathBuf::from("/w"),
            transcript: None,
            moment,
            observed: Vec::new(),
        };

        assert_eq!(
            ClaudeCode.render(
                &event(Moment::Prompt("p".to_owned())),
                Reply::Recall("<m/>")
            ),
            Some("<m/>".to_owned())
        );
        let failure = ClaudeCode
            .render(
                &event(Moment::CommandFailed("e".to_owned())),
                Reply::Recall("<m/>"),
            )
            .expect("failure recall renders");
        let failure: Value = serde_json::from_str(&failure).expect("JSON");
        assert_eq!(
            failure["hookSpecificOutput"]["hookEventName"],
            "PostToolUseFailure"
        );
        assert_eq!(failure["hookSpecificOutput"]["additionalContext"], "<m/>");
        assert_eq!(
            ClaudeCode.render(
                &event(Moment::TurnEnd { continued: false }),
                Reply::Remind("run it")
            ),
            Some(r#"{"decision":"block","reason":"run it"}"#.to_owned())
        );
    }

    #[derive(Debug)]
    struct Config {
        dir: PathBuf,
    }

    impl Config {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-claude-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }

        fn settings(&self) -> Value {
            serde_json::from_str(
                &fs::read_to_string(self.dir.join("settings.json")).expect("settings read"),
            )
            .expect("settings are JSON")
        }
    }

    impl Drop for Config {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn connect_writes_the_hooks_into_settings_and_round_trips() {
        let config = Config::new("connect");
        fs::write(
            config.dir.join("settings.json"),
            r#"{"model":"opus","hooks":{"Stop":[{"hooks":[{"type":"command","command":"afplay done.aiff"}]}]}}"#,
        )
        .expect("settings are writable");
        let program = Program::at("/home/dev/.cargo/bin/trodden");

        Config::apply(ClaudeCode::connect_at(&config.dir, &program).expect("connect"));

        let settings = config.settings();
        assert!(ClaudeCode::connected_at(&config.dir).expect("settings load"));
        assert_eq!(settings["model"], "opus");
        assert_eq!(
            settings["hooks"]["Stop"][0]["hooks"][0]["command"],
            "afplay done.aiff"
        );
        assert_eq!(
            settings["hooks"]["PostToolUseFailure"],
            json!([{"matcher": "Bash", "hooks": [{"type": "command", "command": "/home/dev/.cargo/bin/trodden hook claude-code", "timeout": 5}]}])
        );
        assert!(
            ClaudeCode::connect_at(&config.dir, &program)
                .expect("connect again")
                .is_empty()
        );

        Config::apply(ClaudeCode::disconnect_at(&config.dir).expect("disconnect"));

        assert!(!ClaudeCode::connected_at(&config.dir).expect("settings load"));
        assert_eq!(
            config.settings(),
            json!({"model": "opus", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "afplay done.aiff"}]}]}})
        );
    }

    #[test]
    fn an_installed_plugin_is_not_doubled() {
        let config = Config::new("plugin");
        fs::create_dir_all(config.dir.join("plugins")).expect("plugins dir is writable");
        fs::write(
            config.dir.join("plugins").join("installed_plugins.json"),
            r#"{"plugins":{"trodden@trodden":[{"scope":"user"}]}}"#,
        )
        .expect("plugin list is writable");

        assert!(ClaudeCode::connected_at(&config.dir).expect("settings load"));
        assert!(ClaudeCode::connect_at(&config.dir, &Program::at("/t")).is_err());
        assert!(ClaudeCode::disconnect_at(&config.dir).is_err());
    }
}
