use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::json;
use trodden_capture::{Reminder, qwen::Transcript};
use trodden_core::Trace;
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply, claude_code::ClaudeHookInput};
use crate::connect::{HookFile, Program};

#[derive(Debug)]
pub(crate) struct Qwen;

#[derive(Debug, Default, Deserialize)]
struct QwenHookExtras {
    #[serde(default)]
    submitted_prompt: Option<String>,
}

impl Qwen {
    const SHELL: &[&str] = &["run_shell_command"];

    fn shell_output(error: &str) -> String {
        let Some(start) = error
            .strip_prefix("Output: ")
            .or_else(|| error.split_once("\nOutput: ").map(|(_, rest)| rest))
        else {
            return error.to_owned();
        };
        let end = start
            .rfind("\nError: ")
            .filter(|at| start[*at..].contains("\nExit Code: "))
            .or_else(|| start.rfind("\nExit Code: "))
            .unwrap_or(start.len());
        let output = start[..end].trim();
        if output.is_empty() || output == "(empty)" {
            error.to_owned()
        } else {
            output.to_owned()
        }
    }
}

impl Agent for Qwen {
    fn title(&self) -> &'static str {
        "Qwen Code"
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(Some(QwenHome::locate()?.history()))
    }

    fn transcripts(&self, projects: &Path) -> Result<Vec<PathBuf>> {
        QwenHome::transcripts(projects)
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        Ok((
            Transcript::parse(text, redactor)?,
            Transcript::working_directory(text).map(PathBuf::from),
        ))
    }

    fn event(&self, payload: &str, _redactor: &Redactor) -> Result<Option<HookEvent>> {
        let Some(input) = ClaudeHookInput::parse(payload)? else {
            return Ok(None);
        };
        let extras: QwenHookExtras =
            serde_json::from_str(payload).context("parse the Qwen Code hook payload")?;
        let moment = match input.hook_event_name.as_str() {
            "SessionStart" => Moment::SessionStart,
            "UserPromptSubmit" => match extras.submitted_prompt.as_deref().map(str::trim) {
                Some(prompt) if !prompt.is_empty() && !prompt.starts_with(Reminder::PREFIX) => {
                    Moment::Prompt(prompt.to_owned())
                }
                _ => return Ok(None),
            },
            "PostToolUseFailure" => match input.failure(Self::SHELL) {
                Some(error) => Moment::CommandFailed(Self::shell_output(&error)),
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
            (Moment::Prompt(_), Reply::Recall(envelope)) => Some(ClaudeHookInput::recall_context(
                "UserPromptSubmit",
                envelope,
            )),
            (Moment::CommandFailed(_), Reply::Recall(envelope)) => Some(
                ClaudeHookInput::recall_context("PostToolUseFailure", envelope),
            ),
            (Moment::TurnEnd { .. }, Reply::Remind(reason)) => Some(ClaudeHookInput::block(reason)),
            _ => None,
        }
    }

    fn detected(&self) -> bool {
        QwenHome::locate().is_ok_and(|home| home.config.is_dir())
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        QwenHome::locate()?.connect(program)
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        QwenHome::locate()?.disconnect()
    }

    fn connected(&self) -> Result<bool> {
        QwenHome::locate()?.connected()
    }
}

#[derive(Debug, Clone)]
struct QwenHome {
    config: PathBuf,
    runtime: PathBuf,
}

impl QwenHome {
    const EVENTS: &[(&str, Option<&str>)] = &[
        ("SessionStart", None),
        ("UserPromptSubmit", None),
        ("PostToolUseFailure", Some("^run_shell_command$")),
        ("Stop", None),
        ("PreCompact", None),
        ("SessionEnd", None),
    ];

    const TIMEOUT_SECONDS: u64 = 5;

    fn locate() -> Result<Self> {
        let config = match env::var_os("QWEN_HOME").filter(|dir| !dir.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => env::home_dir()
                .context("find the home directory")?
                .join(".qwen"),
        };
        let runtime = env::var_os("QWEN_RUNTIME_DIR")
            .filter(|dir| !dir.is_empty())
            .map_or_else(|| config.clone(), PathBuf::from);
        Ok(Self { config, runtime })
    }

    #[cfg(test)]
    fn at(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        Self {
            config: dir.clone(),
            runtime: dir,
        }
    }

    fn settings(&self) -> PathBuf {
        self.config.join("settings.json")
    }

    fn history(&self) -> PathBuf {
        self.runtime.join("projects")
    }

    fn transcripts(projects: &Path) -> Result<Vec<PathBuf>> {
        let mut transcripts = Vec::new();
        for project in
            fs::read_dir(projects).with_context(|| format!("list {}", projects.display()))?
        {
            let chats = project
                .context("read a project directory entry")?
                .path()
                .join("chats");
            let Ok(entries) = fs::read_dir(&chats) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "jsonl") {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }

    fn load(&self) -> Result<HookFile> {
        HookFile::load(self.settings()).context("load the Qwen Code settings")
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        let mut file = self.load()?;
        file.remove(&["hooks"], "qwen");
        let command = program.hook("qwen");
        for (event, matcher) in Self::EVENTS {
            let handler = json!({
                "type": "command",
                "name": "trodden",
                "command": command,
                "timeout": Self::TIMEOUT_SECONDS,
            });
            let group = match matcher {
                Some(matcher) => json!({"matcher": matcher, "hooks": [handler]}),
                None => json!({"hooks": [handler]}),
            };
            file.add(&["hooks"], event, group)?;
        }
        Ok(file.change()?.into_iter().collect())
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        if !self.settings().exists() {
            return Ok(Vec::new());
        }
        let mut file = self.load()?;
        file.remove(&["hooks"], "qwen");
        Ok(file.change()?.into_iter().collect())
    }

    fn connected(&self) -> Result<bool> {
        if !self.settings().exists() {
            return Ok(false);
        }
        Ok(self.load()?.contains(&["hooks"], "qwen"))
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use serde_json::Value;

    use super::*;

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-qwen-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }

        fn read(path: &Path) -> Value {
            serde_json::from_str(&fs::read_to_string(path).expect("read")).expect("JSON")
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
        fn moment(payload: Value) -> Option<Moment> {
            let mut base = json!({"session_id": "6f1c2a9e", "transcript_path": "/home/dev/.qwen/projects/-home-dev-shop/chats/6f1c2a9e.jsonl",
                                  "cwd": "/home/dev/shop", "timestamp": "2026-10-09T10:12:03.511Z", "permission_mode": "default"});
            base.as_object_mut()
                .expect("object")
                .extend(payload.as_object().expect("object").clone());
            Qwen.event(&base.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
                .map(|event| event.moment)
        }
    }

    #[test]
    fn only_typed_prompts_recall() {
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "UserPromptSubmit", "prompt": "fix the migration test", "submitted_prompt": "fix the migration test"})
            ),
            Some(Moment::Prompt("fix the migration test".to_owned()))
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "UserPromptSubmit", "prompt": ""})),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "UserPromptSubmit", "prompt": "Trodden: the procedure recalled for this task"})
            ),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "UserPromptSubmit", "prompt": "x", "submitted_prompt": "  "})
            ),
            None
        );
    }

    #[test]
    fn shell_failures_and_turn_ends_map_to_moments() {
        let error = "Command: cargo test -p db\nDirectory: (root)\nOutput: error[E0432]: unresolved import\nError: (none)\nExit Code: 101";
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUseFailure", "tool_name": "run_shell_command", "tool_use_id": "t", "tool_input": {"command": "cargo test -p db"}, "error": error, "is_interrupt": false})
            ),
            Some(Moment::CommandFailed(
                "error[E0432]: unresolved import".to_owned()
            ))
        );
        assert_eq!(
            Qwen::shell_output(
                "Command: x\nDirectory: (root)\nOutput: (empty)\nError: (none)\nExit Code: 1"
            ),
            "Command: x\nDirectory: (root)\nOutput: (empty)\nError: (none)\nExit Code: 1"
        );
        assert_eq!(
            Qwen::shell_output("Output: a\nError: b\nExit Code: 2\nSignal: (none)"),
            "a"
        );
        assert_eq!(Qwen::shell_output("no block"), "no block");
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUseFailure", "tool_name": "run_shell_command", "error": "x", "is_interrupt": true})
            ),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "PostToolUseFailure", "tool_name": "edit", "error": "x"})
            ),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "Stop", "stop_hook_active": false, "last_assistant_message": "done"})
            ),
            Some(Moment::TurnEnd { continued: false })
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "PreCompact", "trigger": "auto"})),
            Some(Moment::Compacting)
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "SessionEnd", "reason": "prompt_input_exit"})
            ),
            Some(Moment::SessionEnd)
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "SessionStart", "source": "startup", "model": "qwen3"})
            ),
            Some(Moment::SessionStart)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "PostToolBatch"})),
            None
        );
    }

    #[test]
    fn replies_are_json() {
        let event = |moment| HookEvent {
            name: String::new(),
            session: "s".to_owned(),
            cwd: PathBuf::from("/w"),
            transcript: None,
            moment,
            observed: Vec::new(),
        };
        let prompt: Value = serde_json::from_str(
            &Qwen
                .render(
                    &event(Moment::Prompt("p".to_owned())),
                    Reply::Recall("<m/>"),
                )
                .expect("renders"),
        )
        .expect("JSON");
        assert_eq!(
            prompt["hookSpecificOutput"]["hookEventName"],
            "UserPromptSubmit"
        );
        assert_eq!(prompt["hookSpecificOutput"]["additionalContext"], "<m/>");
        let failure: Value = serde_json::from_str(
            &Qwen
                .render(
                    &event(Moment::CommandFailed("e".to_owned())),
                    Reply::Recall("<m/>"),
                )
                .expect("renders"),
        )
        .expect("JSON");
        assert_eq!(
            failure["hookSpecificOutput"]["hookEventName"],
            "PostToolUseFailure"
        );
        assert_eq!(
            Qwen.render(
                &event(Moment::TurnEnd { continued: false }),
                Reply::Remind("run it")
            ),
            Some(r#"{"decision":"block","reason":"run it"}"#.to_owned())
        );
    }

    #[test]
    fn connect_round_trips_and_keeps_other_hooks() {
        let scratch = Scratch::new("connect");
        let home = QwenHome::at(scratch.dir.join(".qwen"));
        let program = Program::at("/opt/bin/trodden");

        assert!(!home.connected().expect("no settings yet"));
        assert!(home.disconnect().expect("nothing to remove").is_empty());
        Scratch::apply(home.connect(&program).expect("connect"));
        let mut settings = Scratch::read(&home.settings());
        assert_eq!(
            settings["hooks"]["PostToolUseFailure"][0]["matcher"],
            "^run_shell_command$"
        );
        assert_eq!(settings["hooks"]["Stop"][0]["hooks"][0]["timeout"], 5);
        assert_eq!(
            settings["hooks"]["Stop"][0]["hooks"][0]["command"],
            "/opt/bin/trodden hook qwen"
        );

        settings["model"] = json!({"name": "qwen3-coder-plus"});
        settings["hooks"]["Stop"]
            .as_array_mut()
            .expect("list")
            .insert(
                0,
                json!({"hooks": [{"type": "command", "command": "afplay done.aiff"}]}),
            );
        fs::write(home.settings(), settings.to_string()).expect("written");
        assert!(home.connect(&program).expect("connect again").is_empty());

        Scratch::apply(home.disconnect().expect("disconnect"));
        assert!(!home.connected().expect("settings load"));
        assert_eq!(
            Scratch::read(&home.settings()),
            json!({"model": {"name": "qwen3-coder-plus"}, "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "afplay done.aiff"}]}]}})
        );
    }

    #[test]
    fn transcripts_are_main_session_files() {
        let scratch = Scratch::new("transcripts");
        let projects = scratch.dir.join("projects");
        for path in [
            "-home-dev-shop/chats/6f1c2a9e.jsonl",
            "-home-dev-shop/subagents/6f1c2a9e/agent-1.jsonl",
            "-home-dev-blog/chats/a1.jsonl",
            "-home-dev-blog/chats/a1.jsonl.stream",
        ] {
            let path = projects.join(path);
            fs::create_dir_all(path.parent().expect("parent")).expect("dir");
            fs::write(&path, "{}\n").expect("written");
        }

        let found = QwenHome::transcripts(&projects).expect("list");

        assert_eq!(
            found,
            [
                projects.join("-home-dev-blog/chats/a1.jsonl"),
                projects.join("-home-dev-shop/chats/6f1c2a9e.jsonl")
            ]
        );
    }
}
