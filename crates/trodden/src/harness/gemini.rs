use std::{
    env, fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use trodden_capture::{
    Reminder,
    gemini::{ShellResult, Transcript},
};
use trodden_core::Trace;
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply, claude_code::ClaudeHookInput};
use crate::connect::{HookFile, Program};

#[derive(Debug)]
pub(crate) struct Gemini;

impl Agent for Gemini {
    fn title(&self) -> &'static str {
        "Gemini CLI"
    }

    fn history(&self) -> Result<Option<PathBuf>> {
        Ok(Some(GeminiHome::locate()?.history()))
    }

    fn transcripts(&self, history: &Path) -> Result<Vec<PathBuf>> {
        GeminiHome::transcripts(history)
    }

    fn parse(&self, text: &str, redactor: &Redactor) -> Result<(Trace, Option<PathBuf>)> {
        let cwd = GeminiHome::locate()
            .ok()
            .and_then(|home| home.project_root(text))
            .or_else(|| GeminiHome::added_directory(text));
        let trace = Transcript::parse(text, cwd.as_deref().and_then(Path::to_str), redactor)?;
        Ok((trace, cwd))
    }

    fn event(&self, payload: &str, _redactor: &Redactor) -> Result<Option<HookEvent>> {
        let Some(input) = ClaudeHookInput::parse(payload)? else {
            return Ok(None);
        };
        let moment = match input.hook_event_name.as_str() {
            "SessionStart" => Moment::SessionStart,
            // An AfterAgent block reruns BeforeAgent with the reminder as the prompt.
            "BeforeAgent" => match input.prompt.as_deref().map(str::trim) {
                Some(prompt) if !prompt.is_empty() && !prompt.starts_with(Reminder::PREFIX) => {
                    Moment::Prompt(prompt.to_owned())
                }
                _ => return Ok(None),
            },
            "AfterTool" => match Self::failure(&input) {
                Some(output) => Moment::CommandFailed(output),
                None => return Ok(None),
            },
            "AfterAgent" => Moment::TurnEnd {
                continued: input.stop_hook_active,
            },
            "PreCompress" => Moment::Compacting,
            "SessionEnd" => Moment::SessionEnd,
            _ => return Ok(None),
        };
        Ok(input.event(moment))
    }

    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_), Reply::Recall(envelope)) => {
                Some(ClaudeHookInput::recall_context("BeforeAgent", envelope))
            }
            (Moment::CommandFailed(_), Reply::Recall(envelope)) => {
                Some(ClaudeHookInput::recall_context("AfterTool", envelope))
            }
            (Moment::TurnEnd { .. }, Reply::Remind(reason)) => Some(ClaudeHookInput::block(reason)),
            _ => None,
        }
    }

    fn detected(&self) -> bool {
        GeminiHome::locate().is_ok_and(|home| home.detected())
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        GeminiHome::locate()?.connect(program)
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        GeminiHome::locate()?.disconnect()
    }

    fn connected(&self) -> Result<bool> {
        GeminiHome::locate()?.connected()
    }
}

impl Gemini {
    const SHELL: &str = "run_shell_command";

    fn failure(input: &ClaudeHookInput) -> Option<String> {
        if input.tool_name.as_deref() != Some(Self::SHELL) {
            return None;
        }
        let response = input.tool_response.as_ref()?;
        let content = match response.get("llmContent") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        let shell = ShellResult::parse(&content);
        if shell.interrupted {
            return None;
        }
        let spawn_error = response
            .get("error")
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str);
        match (shell.failed, spawn_error) {
            (true, _) => Some(shell.output),
            (false, Some(message)) => Some(message.to_owned()),
            (false, None) => None,
        }
    }
}

#[derive(Debug, Clone)]
struct GeminiHome {
    dir: PathBuf,
}

impl GeminiHome {
    const EVENTS: &[(&str, Option<&str>)] = &[
        ("SessionStart", None),
        ("BeforeAgent", None),
        ("AfterTool", Some("^run_shell_command$")),
        ("AfterAgent", None),
        ("PreCompress", None),
        ("SessionEnd", None),
    ];

    const TIMEOUT_MS: u64 = 5000;

    fn locate() -> Result<Self> {
        let home = match env::var_os("GEMINI_CLI_HOME").filter(|home| !home.is_empty()) {
            Some(home) => PathBuf::from(home),
            None => env::home_dir().context("find the home directory")?,
        };
        Ok(Self::at(home.join(".gemini")))
    }

    fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn settings(&self) -> PathBuf {
        self.dir.join("settings.json")
    }

    fn history(&self) -> PathBuf {
        self.dir.join("tmp")
    }

    fn detected(&self) -> bool {
        self.settings().is_file() || self.history().is_dir()
    }

    fn transcripts(history: &Path) -> Result<Vec<PathBuf>> {
        let mut transcripts = Vec::new();
        for project in
            fs::read_dir(history).with_context(|| format!("list {}", history.display()))?
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
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if !path.is_file() || !name.starts_with("session-") {
                    continue;
                }
                let jsonl = name.ends_with(".jsonl");
                let legacy = name.ends_with(".json") && !path.with_extension("jsonl").exists();
                if jsonl || legacy {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();
        Ok(transcripts)
    }

    fn project_root(&self, text: &str) -> Option<PathBuf> {
        let session = Transcript::session(text)?;
        let short: String = session.chars().take(8).collect();
        let projects = fs::read_dir(self.history()).ok()?;
        for project in projects.flatten() {
            let project = project.path();
            let Ok(chats) = fs::read_dir(project.join("chats")) else {
                continue;
            };
            let holds = chats.flatten().any(|chat| {
                let name = chat.file_name();
                let name = name.to_string_lossy();
                name.starts_with("session-")
                    && (name.ends_with(&format!("-{short}.jsonl"))
                        || name.ends_with(&format!("-{short}.json")))
                    && Self::session_of(&chat.path()).is_none_or(|id| id == session)
            });
            if !holds {
                continue;
            }
            if let Some(root) = fs::read_to_string(project.join(".project_root"))
                .ok()
                .map(|root| PathBuf::from(root.trim()))
                .filter(|root| root.is_absolute())
                .or_else(|| self.registered(&project))
            {
                return Some(root);
            }
        }
        None
    }

    fn session_of(path: &Path) -> Option<String> {
        let file = fs::File::open(path).ok()?;
        let mut first = String::new();
        BufReader::new(file).read_line(&mut first).ok()?;
        serde_json::from_str::<Value>(&first)
            .ok()?
            .get("sessionId")
            .and_then(Value::as_str)
            .map(str::to_owned)
    }

    fn registered(&self, project: &Path) -> Option<PathBuf> {
        let slug = project.file_name()?.to_str()?;
        let text = fs::read_to_string(self.dir.join("projects.json")).ok()?;
        let registry: Value = serde_json::from_str(&text).ok()?;
        registry
            .get("projects")?
            .as_object()?
            .iter()
            .find(|(_, value)| value.as_str() == Some(slug))
            .map(|(root, _)| PathBuf::from(root))
            .filter(|root| root.is_absolute())
    }

    fn added_directory(text: &str) -> Option<PathBuf> {
        Transcript::directories(text)
            .into_iter()
            .map(PathBuf::from)
            .find(|dir| dir.is_absolute())
    }

    fn load(&self) -> Result<HookFile> {
        HookFile::load(self.settings()).context(
            "load the Gemini CLI settings (Gemini CLI allows comments there, Trodden does not)",
        )
    }

    // No `$` in the command: Gemini CLI expands environment variables in settings.json.
    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        let mut file = self.load()?;
        file.remove(&["hooks"], "gemini");
        let command = program.hook("gemini");
        for (event, matcher) in Self::EVENTS {
            let handler = json!({
                "type": "command",
                "name": "trodden",
                "command": command,
                "timeout": Self::TIMEOUT_MS,
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
        file.remove(&["hooks"], "gemini");
        Ok(file.change()?.into_iter().collect())
    }

    fn connected(&self) -> Result<bool> {
        if !self.settings().exists() {
            return Ok(false);
        }
        Ok(self.load()?.contains(&["hooks"], "gemini"))
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
            let dir = env::temp_dir().join(format!("trodden-gemini-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            Self { dir }
        }

        fn home(&self) -> GeminiHome {
            GeminiHome::at(self.dir.join(".gemini"))
        }

        fn apply(changes: Vec<Change>) {
            for change in changes {
                change.apply().expect("change applies");
            }
        }

        fn write(path: &Path, text: &str) {
            fs::create_dir_all(path.parent().expect("parent")).expect("dir is writable");
            fs::write(path, text).expect("file is writable");
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
            let mut base = json!({"session_id": "8f0c2a3e", "transcript_path": "/home/dev/.gemini/tmp/shop/chats/session-2026-10-09T10-12-8f0c2a3e.jsonl",
                                  "cwd": "/home/dev/shop", "timestamp": "2026-10-09T10:12:31.402Z"});
            base.as_object_mut()
                .expect("object")
                .extend(payload.as_object().expect("object").clone());
            Gemini
                .event(&base.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
                .map(|event| event.moment)
        }

        fn shell(llm_content: &str) -> Option<Moment> {
            Self::moment(
                json!({"hook_event_name": "AfterTool", "tool_name": "run_shell_command",
                "tool_input": {"command": "cargo clippy"}, "tool_response": {"llmContent": llm_content, "returnDisplay": "x"}}),
            )
        }
    }

    #[test]
    fn hook_events_map_to_moments() {
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "BeforeAgent", "prompt": "fix the paging bug"})
            ),
            Some(Moment::Prompt("fix the paging bug".to_owned()))
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "BeforeAgent", "prompt": "Trodden: the procedure recalled for this task is checked with `npm test`, which has not run since your last change."})
            ),
            None
        );
        assert_eq!(
            Payload::shell(
                "<untrusted_context>\nOutput: error: could not compile `shop`\nExit Code: 101\nProcess Group PGID: 5\n</untrusted_context>"
            ),
            Some(Moment::CommandFailed(
                "error: could not compile `shop`".to_owned()
            ))
        );
        assert_eq!(
            Payload::shell(
                "<untrusted_context>\nOutput: ok\nProcess Group PGID: 5\n</untrusted_context>"
            ),
            None
        );
        assert_eq!(
            Payload::shell("Command was cancelled by user before it could complete."),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "AfterTool", "tool_name": "read_file", "tool_response": {"llmContent": "Exit Code: 1"}})
            ),
            None
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "AfterTool", "tool_name": "run_shell_command", "tool_response": {"llmContent": "Output: (empty)", "error": {"message": "spawn bash ENOENT", "type": "shell_execute_error"}}})
            ),
            Some(Moment::CommandFailed("spawn bash ENOENT".to_owned()))
        );
        assert_eq!(
            Payload::moment(
                json!({"hook_event_name": "AfterAgent", "prompt": "p", "prompt_response": "r", "stop_hook_active": true})
            ),
            Some(Moment::TurnEnd { continued: true })
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "PreCompress", "trigger": "auto"})),
            Some(Moment::Compacting)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "SessionEnd", "reason": "exit"})),
            Some(Moment::SessionEnd)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "SessionStart", "source": "startup"})),
            Some(Moment::SessionStart)
        );
        assert_eq!(
            Payload::moment(json!({"hook_event_name": "BeforeModel"})),
            None
        );
    }

    #[test]
    fn empty_transcript_paths_are_no_transcript() {
        let event = Gemini
            .event(
                &json!({"session_id": "s", "transcript_path": "", "cwd": "/home/dev/shop", "hook_event_name": "AfterAgent"}).to_string(),
                &Redactor::with_home("/home/dev"),
            )
            .expect("parses")
            .expect("event");

        assert_eq!(event.transcript, None);
        assert_eq!(event.session, "s");
    }

    #[test]
    fn every_reply_is_json() {
        let event = |moment| HookEvent {
            name: String::new(),
            session: "s".to_owned(),
            cwd: PathBuf::from("/w"),
            transcript: None,
            moment,
            observed: Vec::new(),
        };
        let prompt = Gemini
            .render(
                &event(Moment::Prompt("p".to_owned())),
                Reply::Recall("<m/>"),
            )
            .expect("prompt recall renders");
        let prompt: Value = serde_json::from_str(&prompt).expect("JSON");
        assert_eq!(prompt["hookSpecificOutput"]["hookEventName"], "BeforeAgent");
        assert_eq!(prompt["hookSpecificOutput"]["additionalContext"], "<m/>");

        let failure = Gemini
            .render(
                &event(Moment::CommandFailed("e".to_owned())),
                Reply::Recall("<m/>"),
            )
            .expect("failure recall renders");
        let failure: Value = serde_json::from_str(&failure).expect("JSON");
        assert_eq!(failure["hookSpecificOutput"]["hookEventName"], "AfterTool");

        assert_eq!(
            Gemini.render(
                &event(Moment::TurnEnd { continued: false }),
                Reply::Remind("run it")
            ),
            Some(r#"{"decision":"block","reason":"run it"}"#.to_owned())
        );
        assert_eq!(
            Gemini.render(&event(Moment::SessionEnd), Reply::Recall("<m/>")),
            None
        );
    }

    #[test]
    fn connect_round_trips_and_keeps_other_hooks() {
        let scratch = Scratch::new("connect");
        let home = scratch.home();
        Scratch::write(
            &home.settings(),
            &json!({"theme": "dark", "hooks": {"AfterAgent": [{"hooks": [{"type": "command", "command": "notify-send done"}]}]}}).to_string(),
        );
        let program = Program::at("/opt/bin/trodden");

        assert!(!home.connected().expect("settings load"));
        Scratch::apply(home.connect(&program).expect("connect"));
        assert!(home.connected().expect("settings load"));
        assert!(home.connect(&program).expect("connect again").is_empty());

        let settings: Value =
            serde_json::from_str(&fs::read_to_string(home.settings()).expect("read"))
                .expect("JSON");
        assert_eq!(settings["theme"], "dark");
        assert_eq!(
            settings["hooks"]["AfterTool"][0]["matcher"],
            "^run_shell_command$"
        );
        assert_eq!(
            settings["hooks"]["AfterTool"][0]["hooks"][0]["timeout"],
            5000
        );
        assert_eq!(
            settings["hooks"]["BeforeAgent"][0]["hooks"][0]["command"],
            "/opt/bin/trodden hook gemini"
        );
        assert_eq!(
            settings["hooks"]["AfterAgent"]
                .as_array()
                .expect("list")
                .len(),
            2
        );

        Scratch::apply(home.disconnect().expect("disconnect"));
        assert!(!home.connected().expect("settings load"));
        let settings: Value =
            serde_json::from_str(&fs::read_to_string(home.settings()).expect("read"))
                .expect("JSON");
        assert_eq!(
            settings,
            json!({"theme": "dark", "hooks": {"AfterAgent": [{"hooks": [{"type": "command", "command": "notify-send done"}]}]}})
        );
    }

    #[test]
    fn settings_with_comments_are_refused_clearly() {
        let scratch = Scratch::new("comments");
        let home = scratch.home();
        Scratch::write(
            &home.settings(),
            "{\n  // my theme\n  \"theme\": \"dark\"\n}\n",
        );

        let error = home
            .connect(&Program::at("/opt/bin/trodden"))
            .expect_err("comments are refused");

        let message = format!("{error:#}");
        assert!(message.contains("Gemini CLI settings"), "{message}");
        assert!(message.contains("add the hooks by hand"), "{message}");
    }

    #[test]
    fn sessions_find_their_project_root() {
        let scratch = Scratch::new("root");
        let home = scratch.home();
        let metadata = json!({"sessionId": "8f0c2a3e-6d1b", "projectHash": "h", "startTime": "2026-10-09T10:12:20Z"}).to_string();
        let shop = home.history().join("shop");
        Scratch::write(&shop.join(".project_root"), "/home/dev/shop\n");
        Scratch::write(
            &shop.join("chats/session-2026-10-09T10-12-8f0c2a3e.jsonl"),
            &format!("{metadata}\n"),
        );
        Scratch::write(&shop.join("chats/8f0c2a3e-6d1b/sub.jsonl"), "{}\n");
        let other = home.history().join("blog");
        Scratch::write(
            &other.join("chats/session-2026-10-01T09-00-11111111.json"),
            "{\n}\n",
        );
        Scratch::write(
            &home.dir.join("projects.json"),
            &json!({"projects": {"/home/dev/blog": "blog"}}).to_string(),
        );

        assert_eq!(
            home.project_root(&metadata),
            Some(PathBuf::from("/home/dev/shop"))
        );
        let legacy =
            json!({"sessionId": "11111111-aaaa", "projectHash": "h", "messages": []}).to_string();
        assert_eq!(
            home.project_root(&legacy),
            Some(PathBuf::from("/home/dev/blog"))
        );
        assert_eq!(
            home.project_root(&json!({"sessionId": "99999999", "projectHash": "h"}).to_string()),
            None
        );

        let transcripts = GeminiHome::transcripts(&home.history()).expect("list");
        assert_eq!(transcripts.len(), 2, "{transcripts:?}");
        assert!(
            transcripts
                .iter()
                .all(|path| path.parent().is_some_and(|dir| dir.ends_with("chats")))
        );
    }
}
