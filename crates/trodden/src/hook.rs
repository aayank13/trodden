use std::{
    env,
    ffi::OsStr,
    fs::OpenOptions,
    io::{self, Read, Write},
    process::{Command, ExitCode, Stdio},
};

use anyhow::{Context, Result, bail};
use jiff::Timestamp;
use serde_json::json;
use trodden::{Home, Workspace};
use trodden_capture::{
    ErrorSignature,
    claude_code::{HookInput, Transcript},
};
use trodden_core::{
    Procedure,
    trace::{Event, EventKind, ToolAction, ToolOutcome},
};
use trodden_extract::ProjectChecks;
use trodden_recall::{Decision, Envelope, ErrorQuery, Match, Outcome, Query, Recall};
use trodden_redact::Redactor;
use trodden_store::{Cue, Injection, Patience, Store};

const MAX_INPUT_BYTES: u64 = 1 << 20;

#[derive(Debug)]
pub(crate) struct Hook;

impl Hook {
    pub(crate) fn run(harness: Option<&OsStr>) -> ExitCode {
        if let Err(error) = Self::handle(harness) {
            Self::log(&error);
        }
        ExitCode::SUCCESS
    }

    fn handle(harness: Option<&OsStr>) -> Result<()> {
        if harness.is_none_or(|harness| harness != "claude-code") {
            bail!("unsupported harness {harness:?}");
        }
        let home = Home::locate()?;
        if !home.is_initialized() {
            return Ok(());
        }
        let mut raw = String::new();
        io::stdin()
            .take(MAX_INPUT_BYTES)
            .read_to_string(&mut raw)
            .context("read the hook payload")?;
        let input: HookInput = serde_json::from_str(&raw).context("parse the hook payload")?;

        match input.hook_event_name.as_str() {
            "UserPromptSubmit" => Self::recall(&home, &input),
            "SessionStart" => {
                let store = home.open_store(Patience::Interactive)?;
                Workspace::resolve(&input.cwd, &store).map(drop)
            }
            "PostToolUseFailure" => Self::recall_error(&home, &input),
            "Stop" => {
                Self::spawn_ingest(&input, false)?;
                Self::remind_to_verify(&home, &input)
            }
            "PreCompact" => Self::spawn_ingest(&input, false),
            "SessionEnd" => Self::spawn_ingest(&input, true),
            _ => Ok(()),
        }
    }

    fn recall(home: &Home, input: &HookInput) -> Result<()> {
        let Some(prompt) = input.prompt.as_deref() else {
            return Ok(());
        };
        let store = home.open_store(Patience::Interactive)?;
        if store.paused()? {
            return Ok(());
        }
        let workspace = Workspace::resolve(&input.cwd, &store)?;
        let mut recall = Recall::new(&store, home.semantic());
        let outcome = recall.recall(&Query {
            prompt,
            repo: workspace.repo.as_str(),
            root: &workspace.root,
            session: Some(&input.session_id),
        })?;
        Self::serve(&store, input, outcome, Cue::Prompt, |envelope| {
            envelope.to_owned()
        })
    }

    fn recall_error(home: &Home, input: &HookInput) -> Result<()> {
        let signature = match (&input.tool_name, &input.error) {
            (Some(tool), Some(error)) if tool == "Bash" && !input.is_interrupt => {
                ErrorSignature::of(error, &Redactor::new())
            }
            _ => None,
        };
        let Some(signature) = signature else {
            return Ok(());
        };
        let store = home.open_store(Patience::Interactive)?;
        if store.paused()? {
            return Ok(());
        }
        let workspace = Workspace::resolve(&input.cwd, &store)?;
        let outcome = Recall::new(&store, None).recall_error(&ErrorQuery {
            signature: &signature,
            repo: workspace.repo.as_str(),
            root: &workspace.root,
            session: Some(&input.session_id),
        })?;
        Self::serve(&store, input, outcome, Cue::Failure, |envelope| {
            json!({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUseFailure",
                    "additionalContext": envelope,
                }
            })
            .to_string()
        })
    }

    fn serve(
        store: &Store,
        input: &HookInput,
        outcome: Outcome,
        cue: Cue,
        format: impl Fn(&str) -> String,
    ) -> Result<()> {
        let (chosen, holdout): (Box<Match>, bool) = match outcome.decision {
            Decision::Inject(chosen) => (chosen, false),
            Decision::Withhold(chosen) => (chosen, true),
            Decision::Abstain(_) => return Ok(()),
        };
        let procedure = &chosen.row.procedure;
        if !holdout {
            let mut stdout = io::stdout().lock();
            writeln!(stdout, "{}", format(&Envelope::render(procedure)))
                .context("print the recalled procedure")?;
            stdout.flush().context("flush the recalled procedure")?;
        }
        store.record_injection(&Injection {
            session: input.session_id.clone(),
            procedure: procedure.id.to_string(),
            revision: procedure.revision,
            holdout,
            cue,
            at: Timestamp::now(),
        })
    }

    fn edited_since_check(events: &[Event], injected: Timestamp, check: &str) -> bool {
        let turn = match events
            .iter()
            .rposition(|event| matches!(event.kind, EventKind::Prompt { .. }))
        {
            Some(start) if events[start].at > injected => return false,
            Some(start) => &events[start..],
            None => events,
        };
        let mut edited_since_check = false;
        for event in turn.iter().filter(|event| event.at >= injected) {
            let EventKind::ToolCall(call) = &event.kind else {
                continue;
            };
            if call.action == ToolAction::Edit && call.outcome == ToolOutcome::Succeeded {
                edited_since_check = true;
            } else if call.action == ToolAction::Run
                && call
                    .args
                    .command
                    .as_deref()
                    .is_some_and(|command| ProjectChecks::ran(check, command))
            {
                edited_since_check = false;
            }
        }
        edited_since_check
    }

    fn remind_to_verify(home: &Home, input: &HookInput) -> Result<()> {
        if input.stop_hook_active {
            return Ok(());
        }
        let (Some(transcript), store) = (
            &input.transcript_path,
            home.open_store(Patience::Interactive)?,
        ) else {
            return Ok(());
        };
        if !store.verify_reminder()? || store.paused()? {
            return Ok(());
        }
        let Some(injected) = store
            .injections(Some(&input.session_id))?
            .into_iter()
            .rev()
            .find(|record| !record.injection.holdout)
        else {
            return Ok(());
        };
        let Some(procedure) = store
            .revisions(&injected.injection.procedure)?
            .into_iter()
            .find(|row| row.procedure.revision == injected.injection.revision)
            .map(|row| row.procedure)
        else {
            return Ok(());
        };
        let Some(check) = procedure
            .verify
            .as_ref()
            .map(|verify| verify.command.as_str())
        else {
            return Ok(());
        };

        let text = std::fs::read_to_string(transcript)
            .with_context(|| format!("read {}", transcript.display()))?;
        let trace = Transcript::parse(&text, &Redactor::new()).context("parse the transcript")?;
        let edited_since_check =
            Self::edited_since_check(&trace.events, injected.injection.at, check);
        if edited_since_check {
            let reason = Self::reminder(&procedure, check);
            println!("{}", json!({ "decision": "block", "reason": reason }));
        }
        Ok(())
    }

    fn reminder(procedure: &Procedure, check: &str) -> String {
        let slots: Vec<String> = procedure
            .slots
            .iter()
            .filter(|slot| check.contains(&format!("{{{}}}", slot.name)))
            .map(|slot| {
                let examples: Vec<String> = slot
                    .examples
                    .iter()
                    .take(2)
                    .map(|example| format!("`{example}`"))
                    .collect();
                if examples.is_empty() {
                    format!("{{{}}} is a placeholder", slot.name)
                } else {
                    format!("{{{}}} was {} before", slot.name, examples.join(" or "))
                }
            })
            .collect();
        let fill = if slots.is_empty() {
            String::new()
        } else {
            format!(" ({}; fill in what fits this change)", slots.join("; "))
        };
        format!(
            "Trodden: the procedure recalled for this task is checked with `{check}`{fill}, \
             which has not run since your last change. Run it once before finishing."
        )
    }

    fn spawn_ingest(input: &HookInput, ended: bool) -> Result<()> {
        let Some(transcript) = &input.transcript_path else {
            return Ok(());
        };
        let mut command = Command::new(env::current_exe().context("find the trodden executable")?);
        command.arg("ingest").arg(transcript).arg("--quiet");
        if ended {
            command.arg("--ended");
        }
        command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        command.spawn().context("start a background ingest")?;
        Ok(())
    }

    fn log(error: &anyhow::Error) {
        let Ok(home) = Home::locate() else { return };
        if !home.is_initialized() {
            return;
        }
        if let Ok(mut log) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(home.hook_log())
        {
            let _ = writeln!(log, "{} {error:#}", Timestamp::now());
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use trodden_core::procedure::{Slot, SlotKind};

    use super::*;

    const CWD: &str = "/home/dev/shop";

    #[derive(Default)]
    struct Session {
        lines: Vec<Value>,
    }

    impl Session {
        fn line(&mut self, at: &str, kind: &str, content: Value, result: Option<Value>) {
            let mut line = json!({
                "type": kind, "sessionId": "s", "cwd": CWD, "timestamp": at,
                "message": {"role": kind, "content": content},
            });
            if let Some(result) = result {
                line["toolUseResult"] = result;
            }
            self.lines.push(line);
        }

        fn prompt(mut self, at: &str, text: &str) -> Self {
            self.line(at, "user", json!(text), None);
            self
        }

        fn edit(mut self, at: &str, path: &str) -> Self {
            let id = format!("call{}", self.lines.len());
            let path = format!("{CWD}/{path}");
            self.line(
                at,
                "assistant",
                json!([{"type": "tool_use", "id": id, "name": "Edit", "input": {"file_path": path}}]),
                None,
            );
            self.line(
                at,
                "user",
                json!([{"type": "tool_result", "tool_use_id": id, "content": "updated"}]),
                Some(json!({"filePath": path, "originalFile": "x\n",
                            "structuredPatch": [{"oldStart": 1, "lines": ["-x", "+y"]}]})),
            );
            self
        }

        fn run(mut self, at: &str, command: &str) -> Self {
            let id = format!("call{}", self.lines.len());
            self.line(
                at,
                "assistant",
                json!([{"type": "tool_use", "id": id, "name": "Bash", "input": {"command": command}}]),
                None,
            );
            self.line(
                at,
                "user",
                json!([{"type": "tool_result", "tool_use_id": id, "content": "ok", "is_error": false}]),
                None,
            );
            self
        }

        fn needs_check(&self, injected: &str) -> bool {
            self.needs(injected, "npm test")
        }

        fn needs(&self, injected: &str, check: &str) -> bool {
            let text: String = self.lines.iter().map(|line| format!("{line}\n")).collect();
            let trace = Transcript::parse(&text, &Redactor::with_home("/home/dev"))
                .expect("valid transcript");
            let injected = injected.parse().expect("valid timestamp");
            Hook::edited_since_check(&trace.events, injected, check)
        }
    }

    #[test]
    fn an_injection_from_an_earlier_turn_does_not_ask_for_its_check() {
        let session = Session::default()
            .prompt(
                "2026-09-21T14:00:00.000Z",
                "Fix the paging bug in src/paginate.js",
            )
            .edit("2026-09-21T14:00:05.000Z", "src/paginate.js")
            .run("2026-09-21T14:00:09.000Z", "npm test")
            .prompt("2026-09-21T14:05:00.000Z", "Reword the intro in README.md")
            .edit("2026-09-21T14:05:04.000Z", "README.md");

        assert!(!session.needs_check("2026-09-21T14:00:00.120Z"));
    }

    #[test]
    fn an_injection_in_this_turn_asks_for_its_check_until_it_runs() {
        let edited = Session::default()
            .prompt("2026-09-21T14:00:00.000Z", "Reword the intro in README.md")
            .edit("2026-09-21T14:00:04.000Z", "README.md")
            .prompt(
                "2026-09-21T14:05:00.000Z",
                "Fix the paging bug in src/paginate.js",
            )
            .edit("2026-09-21T14:05:05.000Z", "src/paginate.js");
        let checked = Session::default()
            .prompt(
                "2026-09-21T14:05:00.000Z",
                "Fix the paging bug in src/paginate.js",
            )
            .edit("2026-09-21T14:05:05.000Z", "src/paginate.js")
            .run("2026-09-21T14:05:09.000Z", "npm test");

        assert!(edited.needs_check("2026-09-21T14:05:00.120Z"));
        assert!(edited.needs_check("2026-09-21T14:05:00.000Z"));
        assert!(!checked.needs_check("2026-09-21T14:05:00.120Z"));
    }

    #[test]
    fn a_failure_injection_only_counts_edits_made_after_it() {
        let failed = Session::default()
            .prompt(
                "2026-09-21T14:00:00.000Z",
                "Fix the paging bug in src/paginate.js",
            )
            .edit("2026-09-21T14:00:05.000Z", "src/paginate.js")
            .run("2026-09-21T14:00:09.000Z", "npm run build");
        let fixed = Session::default()
            .prompt(
                "2026-09-21T14:00:00.000Z",
                "Fix the paging bug in src/paginate.js",
            )
            .run("2026-09-21T14:00:09.000Z", "npm run build")
            .edit("2026-09-21T14:00:20.000Z", "src/paginate.js");

        assert!(!failed.needs_check("2026-09-21T14:00:10.000Z"));
        assert!(fixed.needs_check("2026-09-21T14:00:10.000Z"));
    }

    #[test]
    fn a_check_with_a_slot_counts_as_run_with_any_value() {
        let edited = Session::default()
            .prompt(
                "2026-09-21T14:00:00.000Z",
                "due_prints in src/report.rs counts drafts too",
            )
            .edit("2026-09-21T14:00:05.000Z", "src/report.rs");
        let checked = Session::default()
            .prompt(
                "2026-09-21T14:00:00.000Z",
                "due_prints in src/report.rs counts drafts too",
            )
            .edit("2026-09-21T14:00:05.000Z", "src/report.rs")
            .run("2026-09-21T14:00:09.000Z", "cargo test due_prints");

        assert!(edited.needs("2026-09-21T14:00:00.120Z", "cargo test {test}"));
        assert!(!checked.needs("2026-09-21T14:00:00.120Z", "cargo test {test}"));
        assert!(checked.needs("2026-09-21T14:00:00.120Z", "cargo test"));
    }

    #[test]
    fn the_reminder_shows_what_a_slot_stood_for() {
        let mut procedure = Procedure::example();
        procedure.slots = vec![Slot {
            name: "test".to_owned(),
            kind: SlotKind::Identifier,
            examples: vec![
                "count_prints".to_owned(),
                "total_prints".to_owned(),
                "late_prints".to_owned(),
            ],
        }];

        assert_eq!(
            Hook::reminder(&procedure, "cargo test {test}"),
            "Trodden: the procedure recalled for this task is checked with `cargo test {test}` \
             ({test} was `count_prints` or `total_prints` before; fill in what fits this change), \
             which has not run since your last change. Run it once before finishing."
        );
        assert_eq!(
            Hook::reminder(&procedure, "npm test"),
            "Trodden: the procedure recalled for this task is checked with `npm test`, \
             which has not run since your last change. Run it once before finishing."
        );
    }
}
