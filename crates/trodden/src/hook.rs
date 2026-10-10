use std::{
    env,
    ffi::OsStr,
    fmt::Display,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    panic::{self, PanicHookInfo},
    path::Path,
    process::{self, Command, ExitCode, Stdio},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use jiff::Timestamp;
use trodden::{Harness, Home, HookEvent, Moment, Reply, Workspace};
use trodden_capture::{ErrorSignature, Reminder, journal::Observation};
use trodden_core::{
    Procedure,
    trace::{Event, EventKind, ToolAction, ToolOutcome},
};
use trodden_extract::ProjectChecks;
use trodden_recall::{Decision, Envelope, ErrorQuery, Match, Outcome, Query, Recall};
use trodden_redact::Redactor;
use trodden_store::{Cue, Injection, Patience, Store};

use crate::output::Output;

#[derive(Debug)]
pub(crate) struct Hook;

impl Hook {
    const RECORD_BUDGET: Duration = Duration::from_secs(1);
    const RECORD_PAUSE: Duration = Duration::from_millis(20);

    pub(crate) fn run(harness: Option<&OsStr>) -> ExitCode {
        panic::set_hook(Box::new(Self::on_panic));
        if let Err(error) = Self::handle(harness, &mut Output::stdout()) {
            Self::log(&error);
        }
        ExitCode::SUCCESS
    }

    fn on_panic(panic: &PanicHookInfo<'_>) {
        Self::log(panic);
        process::exit(0);
    }

    fn handle(harness: Option<&OsStr>, out: &mut Output<impl Write>) -> Result<()> {
        let Some(harness) = harness.and_then(OsStr::to_str).and_then(Harness::from_name) else {
            bail!("unsupported harness {harness:?}");
        };
        let home = Home::locate()?;
        if !home.is_initialized() {
            return Ok(());
        }
        let raw = Self::read_payload(io::stdin().lock())?;
        let Some(mut event) = harness.event(&raw, &Redactor::new())? else {
            return Ok(());
        };
        if harness.journaled() {
            let journal = home.journal(harness.as_str(), &event.session);
            Self::journal(&home, &journal, &event.observed)?;
            event.transcript = Some(journal);
        }
        Self::respond(harness, &home, &event, out)
    }

    fn read_payload(mut input: impl Read) -> Result<String> {
        let mut raw = String::new();
        input
            .read_to_string(&mut raw)
            .context("read the hook payload")?;
        Ok(raw)
    }

    fn respond(
        harness: Harness,
        home: &Home,
        event: &HookEvent,
        out: &mut Output<impl Write>,
    ) -> Result<()> {
        match &event.moment {
            Moment::SessionStart => {
                let store = home.open_store(Patience::Interactive)?;
                Workspace::resolve_cached(&event.cwd, &store).map(drop)
            }
            Moment::Prompt(prompt) => Self::recall(harness, home, event, prompt, out),
            Moment::CommandFailed(error) => Self::recall_error(harness, home, event, error, out),
            Moment::ToolDone => Ok(()),
            Moment::TurnEnd { continued } => {
                Self::spawn_ingest(harness, event, false)?;
                if *continued {
                    return Ok(());
                }
                Self::remind_to_verify(harness, home, event, out)
            }
            Moment::Compacting => Self::spawn_ingest(harness, event, false),
            Moment::SessionEnd => Self::spawn_ingest(harness, event, true),
        }
    }

    fn journal(home: &Home, journal: &Path, observed: &[Observation]) -> Result<()> {
        if observed.is_empty() || home.open_store(Patience::Interactive)?.paused()? {
            return Ok(());
        }
        let mut lines = String::new();
        for observation in observed {
            lines.push_str(&observation.to_line()?);
        }
        if let Some(dir) = journal.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(journal)
            .and_then(|mut file| file.write_all(lines.as_bytes()))
            .with_context(|| format!("append to {}", journal.display()))
    }

    fn recall(
        harness: Harness,
        home: &Home,
        event: &HookEvent,
        prompt: &str,
        out: &mut Output<impl Write>,
    ) -> Result<()> {
        let store = home.open_store(Patience::Interactive)?;
        if store.paused()? {
            return Ok(());
        }
        let Some(workspace) = Workspace::resolve_cached(&event.cwd, &store)? else {
            return Ok(());
        };
        let outcome = home.recall(&store).recall(&Query {
            prompt,
            repo: workspace.repo.as_str(),
            root: &workspace.root,
            session: Some(&event.session),
        })?;
        if let Some(error) = &outcome.semantic_error {
            Self::log(format_args!("recalled without semantic matching: {error}"));
        }
        Self::serve(harness, &store, event, outcome, Cue::Prompt, out)
    }

    fn recall_error(
        harness: Harness,
        home: &Home,
        event: &HookEvent,
        error: &str,
        out: &mut Output<impl Write>,
    ) -> Result<()> {
        let Some(signature) = ErrorSignature::of(error, &Redactor::new()) else {
            return Ok(());
        };
        let store = home.open_store(Patience::Interactive)?;
        if store.paused()? {
            return Ok(());
        }
        let Some(workspace) = Workspace::resolve_cached(&event.cwd, &store)? else {
            return Ok(());
        };
        let outcome = Recall::new(&store, None).recall_error(&ErrorQuery {
            signature: &signature,
            repo: workspace.repo.as_str(),
            root: &workspace.root,
            session: Some(&event.session),
        })?;
        Self::serve(harness, &store, event, outcome, Cue::Failure, out)
    }

    // Nothing is recorded when the agent cannot show it: an unseen holdout would skew the
    // comparison.
    fn serve(
        harness: Harness,
        store: &Store,
        event: &HookEvent,
        outcome: Outcome,
        cue: Cue,
        out: &mut Output<impl Write>,
    ) -> Result<()> {
        let (chosen, holdout): (Box<Match>, bool) = match outcome.decision {
            Decision::Inject(chosen) => (chosen, false),
            Decision::Withhold(chosen) => (chosen, true),
            Decision::Abstain(_) => return Ok(()),
        };
        let procedure = &chosen.row.procedure;
        let envelope = Envelope::render(procedure);
        let Some(rendered) = harness.render(event, Reply::Recall(&envelope)) else {
            return Ok(());
        };
        let injection = Injection {
            session: event.session.clone(),
            procedure: procedure.id.to_string(),
            revision: procedure.revision,
            holdout,
            cue,
            at: Timestamp::now(),
        };
        Self::retry(Self::RECORD_BUDGET, Store::is_busy, || {
            store.record_injection(&injection)
        })
        .context("record the injection before showing the procedure")?;
        if !holdout {
            writeln!(out, "{rendered}").context("print the recalled procedure")?;
            out.flush().context("flush the recalled procedure")?;
        }
        Ok(())
    }

    fn retry(
        budget: Duration,
        transient: impl Fn(&anyhow::Error) -> bool,
        mut attempt: impl FnMut() -> Result<()>,
    ) -> Result<()> {
        let deadline = Instant::now() + budget;
        loop {
            match attempt() {
                Err(error) if transient(&error) && Instant::now() < deadline => {
                    thread::sleep(Self::RECORD_PAUSE);
                }
                result => return result,
            }
        }
    }

    // Gemini CLI and Qwen Code write the prompt after its hook ran, so it can be stamped after
    // its injection.
    const PROMPT_LAG: jiff::SignedDuration = jiff::SignedDuration::from_secs(10);

    fn edited_since_check(events: &[Event], injected: Timestamp, check: &str) -> bool {
        let turn = match events
            .iter()
            .rposition(|event| matches!(event.kind, EventKind::Prompt { .. }))
        {
            Some(start) if events[start].at > injected + Self::PROMPT_LAG => return false,
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

    // At most one reminder per injection: several agents never say a turn already continued.
    fn remind_to_verify(
        harness: Harness,
        home: &Home,
        event: &HookEvent,
        out: &mut Output<impl Write>,
    ) -> Result<()> {
        let Some(transcript) = &event.transcript else {
            return Ok(());
        };
        let store = home.open_store(Patience::Interactive)?;
        if !store.verify_reminder()? || store.paused()? {
            return Ok(());
        }
        let Some(injected) = store
            .injections(Some(&event.session))?
            .into_iter()
            .rev()
            .find(|record| !record.injection.holdout)
        else {
            return Ok(());
        };
        if injected.reminded {
            return Ok(());
        }
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

        let text =
            Harness::read(transcript).with_context(|| format!("load {}", transcript.display()))?;
        let (trace, _) = harness
            .parse(&text, &Redactor::new())
            .with_context(|| format!("parse {}", transcript.display()))?;
        if !Self::edited_since_check(&trace.events, injected.injection.at, check) {
            return Ok(());
        }
        let reason = Self::reminder(&procedure, check);
        let Some(rendered) = harness.render(event, Reply::Remind(&reason)) else {
            return Ok(());
        };
        store.mark_reminded(injected.rowid, Timestamp::now())?;
        writeln!(out, "{rendered}").context("print the verify reminder")?;
        out.flush().context("flush the verify reminder")?;
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
            "{} for this task is checked with `{check}`{fill}, \
             which has not run since your last change. Run it once before finishing.",
            Reminder::PREFIX
        )
    }

    fn spawn_ingest(harness: Harness, event: &HookEvent, ended: bool) -> Result<()> {
        let Some(transcript) = &event.transcript else {
            return Ok(());
        };
        let mut command = Command::new(env::current_exe().context("find the trodden executable")?);
        command.arg("ingest").arg(transcript).args([
            "--agent",
            harness.as_str(),
            "--quiet",
            "--background",
        ]);
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

    fn log(message: impl Display) {
        if let Ok(home) = Home::locate() {
            home.log_error(message);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, fs, path::PathBuf};

    use serde_json::{Value, json};
    use trodden_capture::claude_code::Transcript;
    use trodden_core::procedure::{Slot, SlotKind};
    use trodden_recall::Signals;
    use trodden_store::ProcedureRow;

    use super::*;

    const CWD: &str = "/home/dev/shop";

    struct Serving {
        store: Store,
        dir: Option<PathBuf>,
    }

    impl Serving {
        fn in_memory() -> Self {
            Self {
                store: Store::open_in_memory().expect("store opens"),
                dir: None,
            }
        }

        fn read_only(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-hook-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch dir is writable");
            let database = dir.join("trodden.db");
            drop(Store::open(&database, Patience::Batch).expect("store opens"));
            Self {
                store: Store::open_read_only(&database, Patience::Interactive)
                    .expect("store opens for reading"),
                dir: Some(dir),
            }
        }

        fn serve(&self, decision: fn(Box<Match>) -> Decision) -> (Result<()>, String) {
            let event = HookEvent {
                name: "UserPromptSubmit".to_owned(),
                session: "s".to_owned(),
                cwd: PathBuf::from(CWD),
                transcript: None,
                moment: Moment::Prompt("Fix the paging bug".to_owned()),
                observed: Vec::new(),
            };
            let chosen = Box::new(Match {
                row: ProcedureRow {
                    rowid: 1,
                    procedure: Procedure::example(),
                },
                signals: Signals::default(),
            });
            let outcome = Outcome {
                decision: decision(chosen),
                candidates: Vec::new(),
                semantic_error: None,
            };
            let mut printed = Vec::new();
            let result = Hook::serve(
                Harness::ClaudeCode,
                &self.store,
                &event,
                outcome,
                Cue::Prompt,
                &mut Output::new(&mut printed),
            );
            (
                result,
                String::from_utf8(printed).expect("printed text is UTF-8"),
            )
        }

        fn recorded(&self) -> Vec<bool> {
            self.store
                .injections(Some("s"))
                .expect("injections read")
                .into_iter()
                .map(|record| record.injection.holdout)
                .collect()
        }
    }

    impl Drop for Serving {
        fn drop(&mut self) {
            if let Some(dir) = &self.dir {
                let _ = fs::remove_dir_all(dir);
            }
        }
    }

    struct Reminding {
        home: Home,
    }

    impl Reminding {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-hook-{name}-{}", process::id()));
            let _ = fs::remove_dir_all(&dir);
            let home = Home::at(dir);
            let mut store = home.initialize().expect("home initializes");
            store.set_verify_reminder(true).expect("setting writes");
            store
                .upsert(&Procedure::example())
                .expect("procedure is stored");
            let stored = store
                .revisions(Procedure::example().id.as_str())
                .expect("revisions read")
                .pop()
                .expect("procedure was stored");
            store
                .record_injection(&Injection {
                    session: "s".to_owned(),
                    procedure: stored.procedure.id.to_string(),
                    revision: stored.procedure.revision,
                    holdout: false,
                    cue: Cue::Prompt,
                    at: "2026-09-21T14:00:00.120Z".parse().expect("valid timestamp"),
                })
                .expect("injection is recorded");
            Self { home }
        }

        fn remind(&self, transcript: &[u8]) -> String {
            let path = self.home.dir().join("transcript.jsonl");
            fs::write(&path, transcript).expect("transcript is writable");
            let event = HookEvent {
                name: "Stop".to_owned(),
                session: "s".to_owned(),
                cwd: PathBuf::from(CWD),
                transcript: Some(path),
                moment: Moment::TurnEnd { continued: false },
                observed: Vec::new(),
            };
            let mut printed = Vec::new();
            Hook::remind_to_verify(
                Harness::ClaudeCode,
                &self.home,
                &event,
                &mut Output::new(&mut printed),
            )
            .expect("reminder runs");
            String::from_utf8(printed).expect("printed text is UTF-8")
        }
    }

    impl Drop for Reminding {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.home.dir());
        }
    }

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
    fn an_injection_is_recorded_and_shown() {
        let serving = Serving::in_memory();

        let (result, printed) = serving.serve(Decision::Inject);

        result.expect("injection is served");
        assert_eq!(
            printed,
            format!("{}\n", Envelope::render(&Procedure::example()))
        );
        assert_eq!(serving.recorded(), [false]);
    }

    #[test]
    fn a_held_out_injection_is_recorded_without_being_shown() {
        let serving = Serving::in_memory();

        let (result, printed) = serving.serve(Decision::Withhold);

        result.expect("holdout is served");
        assert_eq!(printed, "");
        assert_eq!(serving.recorded(), [true]);
    }

    #[test]
    fn an_injection_that_cannot_be_recorded_is_not_shown() {
        let serving = Serving::read_only("unrecorded");
        let started = Instant::now();

        let (result, printed) = serving.serve(Decision::Inject);

        assert!(started.elapsed() < Hook::RECORD_BUDGET);
        let error = result.expect_err("a read-only store cannot record");
        assert!(
            format!("{error:#}").starts_with("record the injection before showing the procedure"),
            "{error:#}"
        );
        assert_eq!(printed, "");
        assert!(serving.recorded().is_empty());
    }

    #[test]
    fn recording_retries_while_the_database_is_busy() {
        let attempts = Cell::new(0);

        let result = Hook::retry(
            Duration::from_secs(1),
            |_| true,
            || {
                attempts.set(attempts.get() + 1);
                if attempts.get() < 4 {
                    bail!("database is locked");
                }
                Ok(())
            },
        );

        result.expect("recording succeeds once the lock is released");
        assert_eq!(attempts.get(), 4);
    }

    #[test]
    fn recording_gives_up_when_its_budget_runs_out() {
        let attempts = Cell::new(0);
        let budget = Duration::from_millis(100);
        let started = Instant::now();

        let result = Hook::retry(
            budget,
            |_| true,
            || {
                attempts.set(attempts.get() + 1);
                bail!("database is locked")
            },
        );

        result.expect_err("the lock is never released");
        assert!(started.elapsed() >= budget);
        assert!(attempts.get() > 1);
    }

    #[test]
    fn recording_fails_at_once_on_errors_that_are_not_busy() {
        let attempts = Cell::new(0);

        let result = Hook::retry(Duration::from_secs(1), Store::is_busy, || {
            attempts.set(attempts.get() + 1);
            bail!("disk I/O error")
        });

        result.expect_err("the error is not transient");
        assert_eq!(attempts.get(), 1);
    }

    #[test]
    fn a_transcript_with_invalid_bytes_still_gets_its_reminder() {
        let reminding = Reminding::new("invalid-bytes");
        let session = Session::default()
            .prompt(
                "2026-09-21T14:00:00.000Z",
                "Fix the paging bug in src/paginate.js",
            )
            .edit("2026-09-21T14:00:05.000Z", "src/paginate.js");
        let mut transcript: Vec<u8> = session
            .lines
            .iter()
            .flat_map(|line| format!("{line}\n").into_bytes())
            .collect();
        transcript.extend(b"{\"type\":\"user\",\"sessionId\":\"s\",\"note\":\"caf\xe9\"}\n");

        let printed = reminding.remind(&transcript);
        let again = reminding.remind(&transcript);

        assert_eq!(again, "", "an injection is reminded once");
        let reminder: Value = serde_json::from_str(&printed).expect("reminder is JSON");
        assert_eq!(reminder["decision"], "block");
        assert!(
            reminder["reason"]
                .as_str()
                .is_some_and(|reason| reason.contains("`npm test`")),
            "{printed}"
        );
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
    fn a_prompt_recorded_after_its_injection_still_starts_its_turn() {
        let session = Session::default()
            .prompt(
                "2026-09-21T14:00:01.500Z",
                "Fix the paging bug in src/paginate.js",
            )
            .edit("2026-09-21T14:00:05.000Z", "src/paginate.js");

        assert!(session.needs_check("2026-09-21T14:00:00.120Z"));
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

    #[test]
    fn a_payload_over_a_mebibyte_is_read_whole() {
        let prompt = format!("Fix the paging bug\n{}", "x".repeat(3 << 20));
        let payload = json!({
            "session_id": "s1",
            "transcript_path": "/home/dev/.claude/projects/shop/s1.jsonl",
            "cwd": CWD,
            "hook_event_name": "UserPromptSubmit",
            "prompt": prompt,
        })
        .to_string();

        let raw = Hook::read_payload(payload.as_bytes()).expect("payload is valid UTF-8");
        let event = Harness::ClaudeCode
            .event(&raw, &Redactor::new())
            .expect("payload parses")
            .expect("prompt is a hook event");

        assert_eq!(raw.len(), payload.len());
        assert!(
            matches!(event.moment, Moment::Prompt(text) if text.starts_with("Fix the paging bug"))
        );
    }
}
