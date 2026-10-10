use std::{
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use jiff::Timestamp;
use serde::Deserialize;
use serde_json::Value;
use trodden_capture::journal::{Edit, Journal, Observation, Observer, Ran};
use trodden_core::{Trace, trace::ToolAction};
use trodden_redact::Redactor;

use super::{Agent, Change, HookEvent, Moment, Reply};
use crate::connect::Program;

#[derive(Debug, Deserialize)]
struct OpenCodeHookInput {
    event: String,
    session: String,
    cwd: PathBuf,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    tool: Option<String>,
    #[serde(default)]
    args: Value,
    #[serde(default)]
    output: Option<String>,
    #[serde(default)]
    exit: Option<i64>,
    #[serde(default)]
    diff: Option<String>,
    #[serde(default)]
    exists: Option<bool>,
}

#[derive(Debug)]
pub(crate) struct OpenCode {
    title: &'static str,
    harness: &'static str,
    app: &'static str,
    notice: &'static str,
}

impl OpenCode {
    pub(crate) const OPENCODE: Self = Self {
        title: "OpenCode",
        harness: "opencode",
        app: "opencode",
        notice: "Restart OpenCode to load the plugin.",
    };

    pub(crate) const KILO: Self = Self {
        title: "Kilo Code",
        harness: "kilo",
        app: "kilo",
        notice: "Restart Kilo Code to load the plugin.",
    };

    const PLUGIN_FILE: &str = "trodden.js";

    const MARKER: &str = "Written by `trodden connect";

    const BOOKKEEPING_TOOLS: &[&str] = &[
        "todowrite",
        "todoread",
        "question",
        "skill",
        "lsp",
        "invalid",
    ];

    const PLUGIN_DIRS: &[&str] = &["plugins", "plugin"];

    // Every hook body is guarded and every call bounded, so a slow or missing Trodden never
    // breaks the agent.
    const PLUGIN: &str = r#"// Written by `trodden connect __AGENT__`; `trodden disconnect __AGENT__` removes it.
import { spawn } from "node:child_process"

const PROGRAM = __PROGRAM__
const AGENT = __AGENT_JSON__
const TIMEOUT_MS = 5000
const REPLY_CHARS = 64 * 1024
const OUTPUT_CHARS = 64 * 1024
const ARG_CHARS = 256 * 1024
const ARGS = ["command", "workdir", "filePath", "oldString", "newString", "content", "patchText", "pattern", "path", "include", "url", "query", "description", "subagent_type"]
const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

let lastTime = 0
let counter = 0

const partId = () => {
  const now = Date.now()
  if (now !== lastTime) {
    lastTime = now
    counter = 0
  }
  counter += 1
  const time = (BigInt(now) * 0x1000n + BigInt(counter)).toString(16).padStart(12, "0").slice(-12)
  let random = ""
  for (let i = 0; i < 14; i += 1) random += BASE62[Math.floor(Math.random() * 62)]
  return "prt_" + time + random
}

const clip = (text, chars, tail) => {
  if (typeof text !== "string" || text.length <= chars) return text
  return tail ? text.slice(-chars) : text.slice(0, chars)
}

const pick = (args) => {
  const picked = {}
  if (!args || typeof args !== "object") return picked
  for (const key of ARGS) {
    if (args[key] !== undefined) picked[key] = clip(args[key], ARG_CHARS, false)
  }
  return picked
}

const run = (payload) =>
  new Promise((resolve) => {
    let child
    let reply = ""
    let done = false
    const finish = (text) => {
      if (done) return
      done = true
      clearTimeout(timer)
      resolve(text)
    }
    const timer = setTimeout(() => {
      try {
        child?.kill()
        child?.stdout?.destroy()
        child?.unref()
      } catch {}
      finish("")
    }, TIMEOUT_MS)
    try {
      child = spawn(PROGRAM, ["hook", AGENT], { stdio: ["pipe", "pipe", "ignore"], windowsHide: true })
      child.on("error", () => finish(""))
      child.stdout.setEncoding("utf8")
      child.stdout.on("data", (chunk) => {
        if (reply.length < REPLY_CHARS) reply += chunk
      })
      child.on("close", (code) => finish(code === 0 ? reply.trim() : ""))
      child.stdin.on("error", () => {})
      child.stdin.end(JSON.stringify(payload))
    } catch {
      finish("")
    }
  })

const guard =
  (hook) =>
  async (...args) => {
    try {
      await hook(...args)
    } catch {}
  }

export const TroddenPlugin = async ({ directory }) => {
  const children = new Set()
  const base = (session) => ({ session, cwd: directory })
  return {
    event: guard(async ({ event }) => {
      const properties = event?.properties ?? {}
      if (event?.type === "session.created") {
        const info = properties.info ?? {}
        if (info.parentID) {
          children.add(info.id ?? properties.sessionID)
          return
        }
        void run({ event: "session_start", ...base(info.id ?? properties.sessionID) })
      } else if (event?.type === "session.status" && properties.status?.type === "idle") {
        if (!children.has(properties.sessionID)) void run({ event: "turn_end", ...base(properties.sessionID) })
      }
    }),
    "chat.message": guard(async (input, output) => {
      if (children.has(input.sessionID)) return
      const prompt = output.parts
        .filter((part) => part.type === "text" && !part.synthetic && !part.ignored)
        .map((part) => part.text)
        .join("\n")
      if (!prompt.trim()) return
      const model = input.model?.modelID ?? output.message?.model?.modelID
      const reply = await run({ event: "prompt", ...base(input.sessionID), prompt, model })
      if (!reply) return
      output.parts.push({
        id: partId(),
        sessionID: input.sessionID,
        messageID: output.message.id,
        type: "text",
        text: reply,
        synthetic: true,
      })
    }),
    "tool.execute.after": guard(async (input, output) => {
      if (children.has(input.sessionID)) return
      const metadata = output?.metadata ?? {}
      const reply = await run({
        event: "tool",
        ...base(input.sessionID),
        tool: input.tool,
        args: pick(input.args),
        output: clip(output?.output, OUTPUT_CHARS, true),
        exit: typeof metadata.exit === "number" ? metadata.exit : null,
        diff: clip(metadata.diff, ARG_CHARS, false),
        exists: typeof metadata.exists === "boolean" ? metadata.exists : undefined,
      })
      if (reply && typeof output.output === "string") output.output += "\n\n" + reply
    }),
    "experimental.session.compacting": guard(async (input) => {
      if (!children.has(input.sessionID)) void run({ event: "compacting", ...base(input.sessionID) })
    }),
  }
}
"#;

    fn config(&self) -> Result<PathBuf> {
        let base = match env::var_os("XDG_CONFIG_HOME").filter(|dir| !dir.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => env::home_dir()
                .context("find the home directory")?
                .join(".config"),
        };
        Ok(base.join(self.app))
    }

    fn data(&self) -> Result<PathBuf> {
        let base = match env::var_os("XDG_DATA_HOME").filter(|dir| !dir.is_empty()) {
            Some(dir) => PathBuf::from(dir),
            None => env::home_dir()
                .context("find the home directory")?
                .join(".local")
                .join("share"),
        };
        Ok(base.join(self.app))
    }

    fn plugin(&self, program: &Program) -> Result<String> {
        let path = serde_json::to_string(&program.path().to_string_lossy())
            .context("encode the trodden path")?;
        let agent = serde_json::to_string(self.harness).context("encode the agent name")?;
        Ok(Self::PLUGIN
            .replace("__PROGRAM__", &path)
            .replace("__AGENT_JSON__", &agent)
            .replace("__AGENT__", self.harness))
    }

    fn connect_at(&self, config: &Path, program: &Program) -> Result<Vec<Change>> {
        let mut changes = self.disconnect_at(config);
        changes.retain(|change| !change.path.starts_with(config.join(Self::PLUGIN_DIRS[0])));
        changes.push(Change::write(
            config.join(Self::PLUGIN_DIRS[0]).join(Self::PLUGIN_FILE),
            self.plugin(program)?,
        ));
        Ok(changes)
    }

    fn disconnect_at(&self, config: &Path) -> Vec<Change> {
        Self::PLUGIN_DIRS
            .iter()
            .map(|dir| config.join(dir).join(Self::PLUGIN_FILE))
            .filter(|path| Self::is_ours(path))
            .map(Change::remove)
            .collect()
    }

    fn connected_at(config: &Path) -> bool {
        Self::PLUGIN_DIRS
            .iter()
            .any(|dir| Self::is_ours(&config.join(dir).join(Self::PLUGIN_FILE)))
    }

    fn is_ours(path: &Path) -> bool {
        fs::read_to_string(path).is_ok_and(|text| text.contains(Self::MARKER))
    }

    fn string<'v>(args: &'v Value, key: &str) -> Option<&'v str> {
        args.get(key)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
    }

    fn observe_tool(input: &OpenCodeHookInput, observer: &Observer<'_>) -> Vec<Observation> {
        let Some(tool) = input.tool.as_deref() else {
            return Vec::new();
        };
        if Self::BOOKKEEPING_TOOLS.contains(&tool) {
            return Vec::new();
        }
        let args = &input.args;
        let path = Self::string(args, "filePath");
        let observation = match tool {
            "bash" => return Vec::new(),
            "edit" | "multiedit" => {
                let Some(path) = path else { return Vec::new() };
                let edit = match (input.diff.as_deref(), Self::string(args, "oldString")) {
                    (Some(diff), _) => Edit::Patched { diff },
                    (None, Some(old)) => Edit::Replaced {
                        old,
                        new: Self::string(args, "newString").unwrap_or_default(),
                    },
                    (None, None) => Edit::Unknown,
                };
                observer.edit(tool, path, edit, true)
            }
            "write" => {
                let Some(path) = path else { return Vec::new() };
                let edit = match (input.exists, Self::string(args, "content")) {
                    (Some(true), _) | (_, None) => Edit::Unknown,
                    (_, Some(content)) => Edit::Created { content },
                };
                observer.edit(tool, path, edit, true)
            }
            "apply_patch" => {
                return Patch::files(Self::string(args, "patchText").unwrap_or_default())
                    .into_iter()
                    .map(|(path, created, body)| {
                        let edit = if created {
                            Edit::Created { content: &body }
                        } else {
                            Edit::Patched { diff: &body }
                        };
                        observer.edit(tool, &path, edit, true)
                    })
                    .collect();
            }
            "read" => match path {
                Some(path) => observer.read(tool, path),
                None => return Vec::new(),
            },
            "grep" | "glob" | "list" | "ls" => observer.search(
                tool,
                Self::string(args, "pattern"),
                Self::string(args, "path"),
            ),
            "webfetch" => observer.fetch(tool, Self::string(args, "url"), None),
            "websearch" | "codesearch" => observer.fetch(tool, None, Self::string(args, "query")),
            "task" => observer.other(tool, ToolAction::Delegate),
            _ => observer.other(tool, ToolAction::Other),
        };
        vec![observation]
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
            } else if let Some((_, created, body)) = files.last_mut() {
                let line = if *created {
                    line.strip_prefix('+').unwrap_or(line)
                } else {
                    line
                };
                body.push_str(line);
                body.push('\n');
            }
        }
        files
    }
}

impl Agent for OpenCode {
    fn title(&self) -> &'static str {
        self.title
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
            Journal::parse(text, self.harness, redactor)?,
            Journal::working_directory(text).map(PathBuf::from),
        ))
    }

    fn event(&self, payload: &str, redactor: &Redactor) -> Result<Option<HookEvent>> {
        let input: OpenCodeHookInput =
            serde_json::from_str(payload).context("parse the hook payload")?;
        if input.session.is_empty() || !input.cwd.is_absolute() {
            return Ok(None);
        }
        let root = input.cwd.to_string_lossy().into_owned();
        let here = Self::string(&input.args, "workdir")
            .filter(|dir| Path::new(dir).is_absolute())
            .map_or_else(|| root.clone(), str::to_owned);
        let observer = Observer {
            session: &input.session,
            cwd: &root,
            at: Timestamp::now(),
            model: input.model.as_deref(),
            redactor,
        };
        let mut observed = Vec::new();
        let moment = match input.event.as_str() {
            "session_start" => Moment::SessionStart,
            "prompt" => {
                let Some(prompt) = input
                    .prompt
                    .as_deref()
                    .filter(|prompt| !prompt.trim().is_empty())
                else {
                    return Ok(None);
                };
                observed.push(observer.prompt(prompt));
                Moment::Prompt(prompt.to_owned())
            }
            "tool" if input.tool.as_deref() == Some("bash") => {
                let output = input.output.as_deref().unwrap_or_default();
                let failed = input.exit.is_some_and(|code| code != 0);
                let shell = Observer {
                    cwd: &here,
                    ..observer
                };
                observed.push(shell.command(
                    "bash",
                    Self::string(&input.args, "command").unwrap_or_default(),
                    Ran {
                        output,
                        exit_code: input.exit.and_then(|code| i32::try_from(code).ok()),
                        failed,
                        interrupted: input.exit.is_none(),
                        duration_ms: None,
                    },
                ));
                if failed {
                    Moment::CommandFailed(output.to_owned())
                } else {
                    Moment::ToolDone
                }
            }
            "tool" => {
                observed.extend(Self::observe_tool(&input, &observer));
                Moment::ToolDone
            }
            "turn_end" => Moment::TurnEnd { continued: false },
            "compacting" => {
                observed.push(observer.compaction(true));
                Moment::Compacting
            }
            _ => return Ok(None),
        };
        Ok(Some(HookEvent {
            name: input.event,
            session: input.session,
            cwd: input.cwd,
            transcript: None,
            moment,
            observed,
        }))
    }

    fn render(&self, event: &HookEvent, reply: Reply<'_>) -> Option<String> {
        match (&event.moment, reply) {
            (Moment::Prompt(_) | Moment::CommandFailed(_), Reply::Recall(envelope)) => {
                Some(envelope.to_owned())
            }
            _ => None,
        }
    }

    fn detected(&self) -> bool {
        self.config().is_ok_and(|config| config.is_dir())
            || self.data().is_ok_and(|data| data.is_dir())
    }

    fn notice(&self) -> Option<&'static str> {
        Some(self.notice)
    }

    fn connect(&self, program: &Program) -> Result<Vec<Change>> {
        self.connect_at(&self.config()?, program)
    }

    fn disconnect(&self) -> Result<Vec<Change>> {
        Ok(self.disconnect_at(&self.config()?))
    }

    fn connected(&self) -> Result<bool> {
        Ok(Self::connected_at(&self.config()?))
    }
}

#[cfg(test)]
mod tests {
    use std::process;

    use serde_json::json;
    use trodden_core::trace::{EventKind, ToolOutcome};

    use super::*;

    const ROOT: &str = "/home/dev/shop";

    #[derive(Debug)]
    struct Payload;

    impl Payload {
        fn event(agent: &OpenCode, payload: Value) -> Option<HookEvent> {
            agent
                .event(&payload.to_string(), &Redactor::with_home("/home/dev"))
                .expect("payload parses")
        }

        fn tool(tool: &str, args: Value, extra: Value) -> Value {
            let mut payload = json!({"event": "tool", "session": "ses_6655", "cwd": ROOT, "tool": tool, "args": args});
            payload
                .as_object_mut()
                .expect("object")
                .extend(extra.as_object().expect("object").clone());
            payload
        }

        fn rendered(moment: Moment, reply: Reply<'_>) -> Option<String> {
            let event = HookEvent {
                name: String::new(),
                session: "ses_6655".to_owned(),
                cwd: PathBuf::from(ROOT),
                transcript: None,
                moment,
                observed: Vec::new(),
            };
            OpenCode::OPENCODE.render(&event, reply)
        }
    }

    #[derive(Debug)]
    struct Scratch {
        dir: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-opencode-{name}-{}", process::id()));
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
    fn prompts_and_failures_are_recalled_as_plain_text() {
        let prompt = Payload::event(
            &OpenCode::OPENCODE,
            json!({"event": "prompt", "session": "ses_6655", "cwd": ROOT,
                   "prompt": "fix the failing cargo test", "model": "claude-sonnet-4-5"}),
        )
        .expect("event");
        let failed = Payload::event(
            &OpenCode::OPENCODE,
            Payload::tool(
                "bash",
                json!({"command": "cargo test", "workdir": ROOT}),
                json!({"output": "error[E0433]: failed to resolve", "exit": 101}),
            ),
        )
        .expect("event");
        let passed = Payload::event(
            &OpenCode::OPENCODE,
            Payload::tool(
                "bash",
                json!({"command": "cargo test"}),
                json!({"output": "ok", "exit": 0}),
            ),
        )
        .expect("event");

        assert_eq!(
            prompt.moment,
            Moment::Prompt("fix the failing cargo test".to_owned())
        );
        assert_eq!(
            prompt.observed[0].model.as_deref(),
            Some("claude-sonnet-4-5")
        );
        assert_eq!(
            failed.moment,
            Moment::CommandFailed("error[E0433]: failed to resolve".to_owned())
        );
        assert_eq!(passed.moment, Moment::ToolDone);
        assert_eq!(
            Payload::rendered(Moment::Prompt("p".to_owned()), Reply::Recall("<m/>")),
            Some("<m/>".to_owned())
        );
        assert_eq!(
            Payload::rendered(Moment::CommandFailed("e".to_owned()), Reply::Recall("<m/>")),
            Some("<m/>".to_owned())
        );
        assert_eq!(
            Payload::rendered(
                Moment::TurnEnd { continued: false },
                Reply::Remind("run it")
            ),
            None
        );
    }

    #[test]
    fn lifecycle_events_map_to_moments() {
        let moment = |event: &str| {
            Payload::event(
                &OpenCode::KILO,
                json!({"event": event, "session": "ses_6655", "cwd": ROOT}),
            )
            .map(|event| event.moment)
        };

        assert_eq!(moment("session_start"), Some(Moment::SessionStart));
        assert_eq!(
            moment("turn_end"),
            Some(Moment::TurnEnd { continued: false })
        );
        assert_eq!(moment("compacting"), Some(Moment::Compacting));
        assert_eq!(moment("prompt"), None);
        assert_eq!(moment("dispose"), None);
        assert_eq!(
            Payload::event(
                &OpenCode::KILO,
                json!({"event": "turn_end", "session": "s", "cwd": "shop"})
            ),
            None
        );
        assert!(
            OpenCode::KILO
                .event("{}", &Redactor::with_home("/home/dev"))
                .is_err()
        );
    }

    #[test]
    fn a_session_journals_into_a_trace() {
        let payloads = [
            json!({"event": "prompt", "session": "ses_6655", "cwd": ROOT, "prompt": "Page 2 repeats the last product. Fix it."}),
            Payload::tool(
                "read",
                json!({"filePath": format!("{ROOT}/src/paginate.js")}),
                json!({"output": "..."}),
            ),
            Payload::tool(
                "edit",
                json!({"filePath": format!("{ROOT}/src/paginate.js"), "oldString": "end + 1", "newString": "end"}),
                json!({"diff": "Index: src/paginate.js\n--- a\n+++ b\n@@ -3,1 +3,1 @@\n-  return items.slice(start, end + 1);\n+  return items.slice(start, end);\n"}),
            ),
            Payload::tool(
                "write",
                json!({"filePath": format!("{ROOT}/CHANGELOG.md"), "content": "# Changes\n- paging\n"}),
                json!({"exists": false}),
            ),
            Payload::tool(
                "apply_patch",
                json!({"patchText": "*** Begin Patch\n*** Add File: src/tax.js\n+export const RATE = 0.2;\n+export default RATE;\n*** Update File: src/cart.js\n@@\n-  return sum;\n+  return sum + tax;\n*** End Patch"}),
                json!({}),
            ),
            Payload::tool("todowrite", json!({}), json!({})),
            Payload::tool(
                "bash",
                json!({"command": "npm test", "workdir": format!("{ROOT}/web")}),
                json!({"output": "1 passing", "exit": 0}),
            ),
            Payload::tool(
                "bash",
                json!({"command": "npm run e2e"}),
                json!({"output": "User aborted the command", "exit": null}),
            ),
            json!({"event": "compacting", "session": "ses_6655", "cwd": ROOT}),
        ];
        let mut journal = String::new();
        for payload in payloads {
            for observation in Payload::event(&OpenCode::OPENCODE, payload)
                .expect("event")
                .observed
            {
                journal.push_str(&observation.to_line().expect("line encodes"));
            }
        }

        let (trace, cwd) = OpenCode::OPENCODE
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
        assert_eq!(trace.harness.as_str(), "opencode");
        assert_eq!(trace.session.as_str(), "ses_6655");
        assert!(
            matches!(&trace.events[0].kind, EventKind::Prompt { summary } if summary == "Page 2 repeats the last product. Fix it.")
        );
        assert_eq!(calls.len(), 7, "{calls:?}");
        assert_eq!(calls[0].action, ToolAction::Read);
        assert_eq!(calls[1].changes[0].path, "src/paginate.js");
        assert_eq!(
            (
                calls[1].changes[0].lines_added,
                calls[1].changes[0].lines_removed
            ),
            (1, 1)
        );
        assert!(calls[2].changes[0].created);
        assert_eq!(calls[2].changes[0].lines_added, 2);
        assert_eq!(calls[3].changes[0].path, "src/tax.js");
        assert!(calls[3].changes[0].created);
        assert_eq!(calls[3].changes[0].lines_added, 2);
        assert_eq!(
            (
                calls[4].changes[0].lines_added,
                calls[4].changes[0].lines_removed
            ),
            (1, 1)
        );
        assert_eq!(calls[5].args.command.as_deref(), Some("cd web && npm test"));
        assert_eq!(calls[5].outcome, ToolOutcome::Succeeded);
        assert_eq!(calls[6].outcome, ToolOutcome::Interrupted);
        assert!(matches!(
            trace.events.last().map(|event| &event.kind),
            Some(EventKind::Compaction { .. })
        ));
    }

    #[test]
    fn the_plugin_names_its_agent_and_program() {
        let program = Program::at("/Users/dev/Application Support/it's/trodden");
        let opencode = OpenCode::OPENCODE.plugin(&program).expect("renders");
        let kilo = OpenCode::KILO.plugin(&program).expect("renders");

        assert!(
            opencode.contains(r#"const PROGRAM = "/Users/dev/Application Support/it's/trodden""#),
            "{opencode}"
        );
        assert!(opencode.contains(r#"const AGENT = "opencode""#));
        assert!(opencode.starts_with("// Written by `trodden connect opencode`"));
        assert!(kilo.contains(r#"const AGENT = "kilo""#));
        assert!(!kilo.contains("__"), "every placeholder is filled");
    }

    #[test]
    fn connect_owns_only_its_plugin_file() {
        let scratch = Scratch::new("connect");
        let program = Program::at("/home/dev/.cargo/bin/trodden");
        let theirs = scratch.dir.join("plugin").join("trodden.js");
        fs::create_dir_all(theirs.parent().expect("parent")).expect("writable");
        fs::write(&theirs, "export const Mine = async () => ({})\n").expect("writable");
        let legacy = scratch.dir.join("plugin").join("other.js");
        fs::write(&legacy, "export const Other = async () => ({})\n").expect("writable");

        assert!(!OpenCode::connected_at(&scratch.dir));
        Scratch::apply(
            OpenCode::KILO
                .connect_at(&scratch.dir, &program)
                .expect("connects"),
        );
        let ours = scratch.dir.join("plugins").join("trodden.js");
        assert!(OpenCode::connected_at(&scratch.dir));
        assert!(
            OpenCode::KILO
                .connect_at(&scratch.dir, &program)
                .expect("connects")
                .iter()
                .all(|change| !change.is_needed()),
            "idempotent"
        );
        assert!(
            fs::read_to_string(&ours)
                .expect("written")
                .contains(r#"const AGENT = "kilo""#)
        );

        Scratch::apply(OpenCode::KILO.disconnect_at(&scratch.dir));
        assert!(!ours.exists());
        assert!(
            theirs.exists() && legacy.exists(),
            "files Trodden did not write stay"
        );
        assert!(!OpenCode::connected_at(&scratch.dir));
    }
}
