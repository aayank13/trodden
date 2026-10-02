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
use trodden_core::trace::{EventKind, ToolAction, ToolOutcome};
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
                ErrorSignature::of(error)
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
        let check = store
            .revisions(&injected.injection.procedure)?
            .into_iter()
            .find(|row| row.procedure.revision == injected.injection.revision)
            .and_then(|row| row.procedure.verify)
            .map(|verify| verify.command);
        let Some(check) = check else {
            return Ok(());
        };

        let text = std::fs::read_to_string(transcript)
            .with_context(|| format!("read {}", transcript.display()))?;
        let trace = Transcript::parse(&text, &Redactor::new()).context("parse the transcript")?;
        let turn = trace
            .events
            .iter()
            .rposition(|event| matches!(event.kind, EventKind::Prompt { .. }))
            .map_or(&trace.events[..], |start| &trace.events[start..]);
        let mut edited_since_check = false;
        for event in turn
            .iter()
            .filter(|event| event.at >= injected.injection.at)
        {
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
                    .is_some_and(|command| ProjectChecks::same(command, &check))
            {
                edited_since_check = false;
            }
        }
        if edited_since_check {
            let reason = format!(
                "Trodden: the procedure recalled for this task is checked with `{check}`, \
                 which has not run since your last change. Run it once before finishing."
            );
            println!("{}", json!({ "decision": "block", "reason": reason }));
        }
        Ok(())
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
