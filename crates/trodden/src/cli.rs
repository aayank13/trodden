use std::{
    env,
    ffi::{OsStr, OsString},
    fs::{self, File, TryLockError},
    io::Write,
    iter,
    path::PathBuf,
    process::{Command as Process, ExitCode},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use trodden::{Harness, Home, Ingest, IngestReport, Workspace};
use trodden_core::RepoId;
use trodden_embed::ModelPack;
use trodden_recall::{Abstention, Decision, Envelope, Match, Outcome, Query};
use trodden_store::{Forget, Patience, Store};

use crate::{
    mcp::Server,
    output::{Closed, Output},
};

#[derive(Debug, Parser)]
#[command(name = "trodden", version, about)]
pub(crate) struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    #[command(about = "Create the data directory and start capturing sessions")]
    Init,
    #[command(about = "Show what is stored and whether capture is running")]
    Status,
    #[command(about = "Check the installation for problems")]
    Doctor,
    #[command(about = "List procedures learned in this repository")]
    List {
        #[arg(long, help = "List procedures from every repository")]
        all_repos: bool,
        #[arg(
            long,
            help = "Include every revision: candidates, archived, quarantined and retired"
        )]
        revisions: bool,
    },
    #[command(about = "Show a procedure: what an agent would see, and the full record")]
    Show {
        #[arg(help = "Procedure id, such as `p_5cf7b08af57c`")]
        id: String,
    },
    #[command(about = "Show what recall would inject for a prompt")]
    Recall {
        #[arg(help = "The prompt")]
        prompt: String,
        #[arg(long, help = "Directory the prompt is submitted in [default: current]")]
        cwd: Option<PathBuf>,
        #[arg(long, help = "Show every candidate and how it scored")]
        explain: bool,
    },
    #[command(about = "List recent tasks that did not become procedures, and why")]
    Rejected {
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    #[command(about = "Show how tasks went with and without each recalled procedure")]
    Outcomes,
    #[command(
        about = "Remove a procedure from recall for good. It stays listed, and later sessions of the same kind are not learned as new procedures"
    )]
    Retire {
        #[arg(help = "Procedure id")]
        id: String,
    },
    #[command(about = "Show or change settings")]
    Config {
        #[command(subcommand)]
        setting: Option<Setting>,
    },
    #[command(about = "Stop capturing sessions and recalling procedures")]
    Pause,
    #[command(about = "Resume capture and recall")]
    Resume,
    #[command(about = "Delete procedures")]
    Forget {
        #[arg(
            conflicts_with_all = ["repo", "all"],
            required_unless_present_any = ["repo", "all"],
            help = "Procedure id to delete"
        )]
        id: Option<String>,
        #[arg(long, help = "Delete every procedure learned in this repository")]
        repo: bool,
        #[arg(
            long,
            requires = "yes",
            help = "Delete everything, including ingest history"
        )]
        all: bool,
        #[arg(long, help = "Confirm deleting everything")]
        yes: bool,
    },
    #[command(about = "Learn from existing Claude Code transcripts")]
    Backfill {
        #[arg(
            long,
            help = "Claude Code's projects directory [default: ~/.claude/projects]"
        )]
        projects: Option<PathBuf>,
    },
    #[command(about = "Learn from one transcript. Hooks run this in the background")]
    Ingest {
        #[arg(help = "Claude Code transcript (`.jsonl`)")]
        transcript: PathBuf,
        #[arg(long, help = "The session has ended, so its last task is complete")]
        ended: bool,
        #[arg(long, help = "Print nothing")]
        quiet: bool,
        #[arg(
            long,
            hide = true,
            help = "Leave the transcript to an ingest that is already running instead of waiting for it"
        )]
        background: bool,
    },
    #[command(about = "Manage the embedding model used for semantic matching")]
    Embeddings {
        #[command(subcommand)]
        command: EmbeddingsCommand,
    },
    #[command(about = "Serve read-only MCP tools over stdio")]
    Mcp,
    #[command(about = "Handle a harness hook (used by the Claude Code plugin)")]
    Hook {
        #[arg(help = "Harness name")]
        harness: Harness,
    },
}

#[derive(Debug, Subcommand)]
enum Setting {
    #[command(
        about = "Share of confident matches withheld to measure the baseline: a number from 0 to 1, or `auto` for 10% dropping to 2% once a procedure's effect is known"
    )]
    Holdout { rate: String },
    #[command(
        about = "Whether the Stop hook asks the agent to run a recalled procedure's check when it changed files without running it: `on` or `off`"
    )]
    VerifyReminder { state: String },
}

#[derive(Debug, Subcommand)]
enum EmbeddingsCommand {
    #[command(about = "Download or import the model and re-embed stored procedures")]
    Install {
        #[arg(
            long,
            help = "Import from a directory holding `tokenizer.json` and `model.safetensors` instead of downloading"
        )]
        from: Option<PathBuf>,
    },
}

#[derive(Debug)]
enum Finding {
    Healthy(String),
    Note(String),
    Problem(anyhow::Error),
}

#[derive(Debug)]
struct ExecutableSearch {
    dirs: Vec<PathBuf>,
    extensions: Vec<OsString>,
}

impl Cli {
    pub(crate) fn run() -> ExitCode {
        let cli = Self::parse();
        let mut out = Output::stdout();
        let result = cli.command.execute(&mut out).and_then(|()| out.flush());
        Self::exit(result, &mut Output::stderr())
    }

    fn exit(result: Result<()>, err: &mut Output<impl Write>) -> ExitCode {
        match result {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) if Closed::caused(&error) => ExitCode::SUCCESS,
            Err(error) => {
                let _ = writeln!(err, "trodden: {error:#}");
                ExitCode::FAILURE
            }
        }
    }
}

impl Command {
    fn execute(self, out: &mut Output<impl Write>) -> Result<()> {
        let home = Home::locate()?;
        if !matches!(self, Self::Init | Self::Doctor | Self::Hook { .. }) {
            ensure!(home.is_initialized(), "run `trodden init` first");
        }
        match self {
            Self::Init => Self::init(&home, out),
            Self::Status => Self::status(&home, out),
            Self::Doctor => Self::doctor(&home, out),
            Self::List {
                all_repos,
                revisions,
            } => Self::list(&home, out, all_repos, revisions),
            Self::Show { id } => Self::show(&home, out, &id),
            Self::Recall {
                prompt,
                cwd,
                explain,
            } => Self::recall(&home, out, &prompt, cwd, explain),
            Self::Rejected { limit } => Self::rejected(&home, out, limit),
            Self::Outcomes => Self::outcomes(&home, out),
            Self::Retire { id } => Self::retire(&home, out, &id),
            Self::Config { setting } => Self::config(&home, out, setting),
            Self::Pause => home.open_store(Patience::Batch)?.set_paused(true),
            Self::Resume => home.open_store(Patience::Batch)?.set_paused(false),
            Self::Forget { id, repo, all, .. } => {
                Self::forget(&home, out, id.as_deref(), repo, all)
            }
            Self::Backfill { projects } => Self::backfill(&home, out, projects),
            Self::Ingest {
                transcript,
                ended,
                quiet,
                background,
            } => {
                let report = if background {
                    Ingest::run_or_defer(&home, Harness::ClaudeCode, &transcript, ended)?
                } else {
                    Some(Ingest::run(&home, Harness::ClaudeCode, &transcript, ended)?)
                };
                if let Some(report) = report.filter(|_| !quiet) {
                    Self::print_report(out, &report)?;
                }
                Ok(())
            }
            Self::Embeddings {
                command: EmbeddingsCommand::Install { from },
            } => Self::install_embeddings(&home, out, from),
            Self::Mcp => Server::serve(home),
            Self::Hook { .. } => {
                bail!("hooks read their payload from stdin; see `trodden help hook`")
            }
        }
    }

    fn init(home: &Home, out: &mut Output<impl Write>) -> Result<()> {
        let fresh = !home.is_initialized();
        home.initialize()?;
        writeln!(out, "Data directory: {}", home.dir().display())?;
        if fresh {
            writeln!(
                out,
                "\nTrodden now learns from Claude Code sessions in this account. It stores\n\
                 the first line of each prompt (up to 200 characters), commands, file paths,\n\
                 touched function names, exit codes and one normalized line per error, with\n\
                 secrets redacted. It never stores file contents, other tool output or the\n\
                 rest of a prompt, and makes no network calls. `trodden pause` stops it;\n\
                 `trodden forget` deletes."
            )?;
        }
        writeln!(out, "\nNext steps:")?;
        writeln!(out, "  1. Install the Claude Code plugin:")?;
        writeln!(out, "       claude plugin marketplace add aayank13/trodden")?;
        writeln!(out, "       claude plugin install trodden@trodden")?;
        if home.embedder().is_none() {
            writeln!(
                out,
                "  2. Optional: enable semantic matching (one-time 32 MB download):"
            )?;
            writeln!(out, "       trodden embeddings install")?;
        }
        writeln!(
            out,
            "  {}. Optional: learn from past sessions:",
            if home.embedder().is_none() { 3 } else { 2 }
        )?;
        writeln!(out, "       trodden backfill")?;
        Ok(())
    }

    fn status(home: &Home, out: &mut Output<impl Write>) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let stats = store.stats()?;
        writeln!(out, "Data directory     {}", home.dir().display())?;
        writeln!(
            out,
            "Capture and recall {}",
            if store.paused()? { "paused" } else { "on" }
        )?;
        writeln!(
            out,
            "Semantic matching  {}",
            match home.open_semantic() {
                Ok(Some(_)) => "on",
                Ok(None) => "off (run `trodden embeddings install`)",
                Err(_) => "off (the model or index is unreadable; run `trodden doctor`)",
            }
        )?;
        writeln!(
            out,
            "Procedures         {} ({} revisions)",
            stats.procedures, stats.revisions
        )?;
        writeln!(out, "Sessions ingested  {}", stats.sessions)?;
        writeln!(out, "Rejected tasks     {}", stats.rejections)?;
        writeln!(out, "Injections         {}", stats.injections)?;
        writeln!(
            out,
            "Holdout            {}",
            store
                .holdout_rate()?
                .map_or_else(|| "auto".to_owned(), |rate| format!("{:.0}%", rate * 100.0))
        )?;
        Ok(())
    }

    fn doctor(home: &Home, out: &mut Output<impl Write>) -> Result<()> {
        let mut problems = 0;
        let mut report = |label: &str, finding: Finding| match finding {
            Finding::Healthy(detail) => writeln!(out, "ok       {label}: {detail}"),
            Finding::Note(detail) => writeln!(out, "note     {label}: {detail}"),
            Finding::Problem(error) => {
                problems += 1;
                writeln!(out, "PROBLEM  {label}: {error:#}")
            }
        };
        report(
            "data directory",
            if home.is_initialized() {
                Finding::Healthy(home.dir().display().to_string())
            } else {
                Finding::Problem(anyhow::anyhow!("not initialized; run `trodden init`"))
            },
        )?;
        if home.is_initialized() {
            report(
                "database",
                home.open_store(Patience::Batch)
                    .and_then(|store| store.stats())
                    .map(|stats| format!("{} procedures", stats.procedures))
                    .into(),
            )?;
        }
        report("embedding model", Finding::embedding_model(home))?;
        report(
            "recall index",
            home.open_index()
                .map(|index| {
                    if index.is_some() {
                        "readable".to_owned()
                    } else {
                        "not built yet".to_owned()
                    }
                })
                .into(),
        )?;
        for program in ["trodden", "git", "claude"] {
            report(
                &format!("`{program}` on PATH"),
                ExecutableSearch::from_env()
                    .and_then(|search| search.find(program))
                    .map(|path| path.display().to_string())
                    .into(),
            )?;
        }
        if let Ok(log) = fs::read_to_string(home.hook_log()) {
            for line in log.lines().rev().take(3) {
                writeln!(out, "note     recent hook error: {line}")?;
            }
        }
        ensure!(problems == 0, "{problems} problem(s) found");
        Ok(())
    }

    fn list(
        home: &Home,
        out: &mut Output<impl Write>,
        all_repos: bool,
        revisions: bool,
    ) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let repo = if all_repos {
            None
        } else {
            Some(Self::workspace(&store, None)?.repo)
        };
        let rows = store.list(repo.as_ref().map(RepoId::as_str), revisions)?;
        if rows.is_empty() {
            writeln!(
                out,
                "No procedures yet{}.",
                if all_repos { "" } else { " in this repository" }
            )?;
            return Ok(());
        }
        writeln!(
            out,
            "{:<16} {:>3}  {:<11} {:>8} {:>7}  TITLE",
            "ID", "REV", "STATE", "SESSIONS", "WORKED"
        )?;
        for row in rows {
            let procedure = row.procedure;
            let state = serde_json::to_value(procedure.state).context("name a lifecycle state")?;
            let judged = procedure.outcomes.successes + procedure.outcomes.failures;
            let worked = if judged == 0 {
                "-".to_owned()
            } else {
                format!("{}/{judged}", procedure.outcomes.successes)
            };
            writeln!(
                out,
                "{:<16} {:>3}  {:<11} {:>8} {:>7}  {}",
                procedure.id,
                procedure.revision,
                state.as_str().unwrap_or_default(),
                procedure.provenance.sources.len(),
                worked,
                procedure.title
            )?;
        }
        Ok(())
    }

    fn show(home: &Home, out: &mut Output<impl Write>, id: &str) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let revisions = store.revisions(id)?;
        ensure!(!revisions.is_empty(), "no procedure {id}");
        for row in revisions {
            writeln!(out, "{}\n", Envelope::render(&row.procedure))?;
            writeln!(
                out,
                "{}\n",
                serde_json::to_string_pretty(&row.procedure).context("serialize a procedure")?
            )?;
        }
        Ok(())
    }

    fn recall(
        home: &Home,
        out: &mut Output<impl Write>,
        prompt: &str,
        cwd: Option<PathBuf>,
        explain: bool,
    ) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let workspace = Self::workspace(&store, cwd)?;
        let outcome = home.recall(&store).recall(&Query {
            prompt,
            repo: workspace.repo.as_str(),
            root: &workspace.root,
            session: None,
        })?;
        if explain {
            Self::print_explanation(out, &outcome)?;
        }
        match outcome.decision {
            Decision::Inject(chosen) | Decision::Withhold(chosen) => {
                writeln!(out, "{}", Envelope::render(&chosen.row.procedure))?;
            }
            Decision::Abstain(reason) => {
                writeln!(out, "Nothing to inject: {}.", Self::describe(&reason))?;
            }
        }
        Ok(())
    }

    fn print_explanation(out: &mut Output<impl Write>, outcome: &Outcome) -> Result<()> {
        if let Some(error) = &outcome.semantic_error {
            writeln!(out, "Recalled without semantic matching: {error}\n")?;
        }
        if outcome.candidates.is_empty() {
            return Ok(());
        }
        writeln!(
            out,
            "{:<16} {:>7} {:>9} {:>9}  {:<24} TITLE",
            "ID", "FUSED", "LEXICAL", "COSINE", "EXACT"
        )?;
        for Match { row, signals } in &outcome.candidates {
            let lexical = signals.lexical.map_or_else(
                || "-".to_owned(),
                |(rank, score)| format!("#{} {score:.1}", rank + 1),
            );
            let cosine = signals.semantic.map_or_else(
                || "-".to_owned(),
                |(rank, cosine)| format!("#{} {cosine:.2}", rank + 1),
            );
            let exact: Vec<&str> = signals.exact.iter().map(|(key, _)| key.as_str()).collect();
            writeln!(
                out,
                "{:<16} {:>7.4} {:>9} {:>9}  {:<24} {}",
                row.procedure.id,
                signals.fused,
                lexical,
                cosine,
                exact.join(","),
                row.procedure.title
            )?;
        }
        writeln!(out)?;
        Ok(())
    }

    fn describe(reason: &Abstention) -> String {
        match reason {
            Abstention::NoCandidates => "no procedure matched".to_owned(),
            Abstention::NotConfident => "the best match was not close enough".to_owned(),
            Abstention::Ambiguous => "two procedures matched about equally".to_owned(),
            Abstention::PreconditionFailed(why) => {
                format!("the best match no longer applies ({why})")
            }
            Abstention::AlreadyInjected => {
                "the best match was already injected this session".to_owned()
            }
            _ => "recall abstained".to_owned(),
        }
    }

    fn outcomes(home: &Home, out: &mut Output<impl Write>) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let summaries = store.outcome_summaries()?;
        if summaries.is_empty() {
            writeln!(out, "No procedure has been recalled yet.")?;
            return Ok(());
        }
        writeln!(
            out,
            "{:<16} {:>10} {:>10} {:>14}  TITLE",
            "ID", "INJECTED", "HELD OUT", "TOOL CALLS"
        )?;
        let rate = |evidence: trodden_learn::Evidence| {
            if evidence.total() == 0 {
                "-".to_owned()
            } else {
                format!("{}/{}", evidence.successes, evidence.total())
            }
        };
        for summary in summaries {
            let calls = match (summary.injected_tool_calls, summary.held_out_tool_calls) {
                (Some(injected), Some(held)) => format!("{injected:.0} vs {held:.0}"),
                (Some(injected), None) => format!("{injected:.0} vs -"),
                _ => "-".to_owned(),
            };
            writeln!(
                out,
                "{:<16} {:>10} {:>10} {:>14}  {}",
                summary.procedure,
                rate(summary.injected),
                rate(summary.held_out),
                calls,
                summary.title
            )?;
        }
        writeln!(
            out,
            "\nINJECTED and HELD OUT count tasks that ended with a passing check, out of those\n\
             that changed files. TOOL CALLS compares the mean effort of those tasks."
        )?;
        Ok(())
    }

    fn retire(home: &Home, out: &mut Output<impl Write>, id: &str) -> Result<()> {
        let ingest = Self::wait_for_ingest(home)?;
        let retired = home.open_store(Patience::Batch)?.retire(id)?;
        ensure!(retired > 0, "no procedure {id}");
        ingest.rebuild_index()?;
        writeln!(out, "Retired {retired} revision(s) of {id}.")?;
        Ok(())
    }

    fn config(home: &Home, out: &mut Output<impl Write>, setting: Option<Setting>) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        match setting {
            Some(Setting::Holdout { rate }) if rate == "auto" => store.set_holdout_rate(None)?,
            Some(Setting::Holdout { rate }) => {
                let rate: f64 = rate
                    .parse()
                    .context("read the rate: a number from 0 to 1, or `auto`")?;
                ensure!(
                    (0.0..=1.0).contains(&rate),
                    "the holdout rate must be from 0 to 1"
                );
                store.set_holdout_rate(Some(rate))?;
            }
            Some(Setting::VerifyReminder { state }) => match state.as_str() {
                "on" => store.set_verify_reminder(true)?,
                "off" => store.set_verify_reminder(false)?,
                other => bail!("expected `on` or `off`, not `{other}`"),
            },
            None => {}
        }
        let holdout = store.holdout_rate()?.map_or_else(
            || "auto (10%, then 2% once a procedure's effect is known)".to_owned(),
            |rate| format!("{:.0}%", rate * 100.0),
        );
        writeln!(out, "holdout          {holdout}")?;
        writeln!(
            out,
            "verify-reminder  {}",
            if store.verify_reminder()? {
                "on"
            } else {
                "off"
            }
        )?;
        Ok(())
    }

    fn rejected(home: &Home, out: &mut Output<impl Write>, limit: usize) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let records = store.rejections(limit)?;
        if records.is_empty() {
            writeln!(out, "No rejected tasks.")?;
        }
        for record in records {
            writeln!(
                out,
                "{}  {}\n    {}",
                record.at,
                record.summary,
                record.rejection.unwrap_or_default()
            )?;
        }
        Ok(())
    }

    fn forget(
        home: &Home,
        out: &mut Output<impl Write>,
        id: Option<&str>,
        repo: bool,
        all: bool,
    ) -> Result<()> {
        let ingest = Self::wait_for_ingest(home)?;
        let mut store = home.open_store(Patience::Batch)?;
        let workspace = if repo {
            Some(Self::workspace(&store, None)?)
        } else {
            None
        };
        let target = match (id, &workspace, all) {
            (_, _, true) => Forget::All,
            (_, Some(workspace), _) => Forget::Repo(workspace.repo.as_str()),
            (Some(id), _, _) => Forget::Procedure(id),
            (None, None, false) => bail!("say what to forget: an id, --repo or --all"),
        };
        let deleted = store.forget(target)?;
        if let Forget::Procedure(id) = target {
            ensure!(deleted > 0, "no procedure {id}");
        }
        let log = home.hook_log();
        if matches!(target, Forget::All) && log.exists() {
            fs::remove_file(&log).with_context(|| format!("remove {}", log.display()))?;
        }
        ingest.rebuild_index()?;
        writeln!(out, "Deleted {deleted} revision(s).")?;
        Ok(())
    }

    fn wait_for_ingest(home: &Home) -> Result<Ingest> {
        let lock = File::create(home.ingest_lock()).context("create the ingest lock")?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => {
                writeln!(
                    Output::stderr(),
                    "Waiting for a running ingest to finish..."
                )?;
            }
            Err(TryLockError::Error(error)) => {
                return Err(error).context("check the ingest lock");
            }
        }
        drop(lock);
        Ingest::start(home)
    }

    fn backfill(
        home: &Home,
        out: &mut Output<impl Write>,
        projects: Option<PathBuf>,
    ) -> Result<()> {
        let harness = Harness::ClaudeCode;
        let history = match projects {
            Some(projects) => projects,
            None => harness.history_dir()?,
        };
        let (report, failures) = Ingest::start(home)?.backfill(harness, &history)?;
        Self::print_report(out, &report)?;
        let mut err = Output::stderr();
        for (transcript, error) in &failures {
            writeln!(err, "skipped {}: {error}", transcript.display())?;
        }
        Ok(())
    }

    fn print_report(out: &mut Output<impl Write>, report: &IngestReport) -> Result<()> {
        writeln!(
            out,
            "{} session(s), {} task(s): {} new procedure(s), {} confirmed, {} new revision(s), {} rejected",
            report.sessions,
            report.tasks,
            report.created,
            report.refreshed,
            report.revised,
            report.rejected.values().sum::<usize>()
        )?;
        for (reason, count) in &report.rejected {
            writeln!(out, "  {count:>4}  {reason}")?;
        }
        if report.settled + report.promoted + report.quarantined + report.aged > 0 {
            writeln!(
                out,
                "{} injection(s) judged; {} revision(s) promoted, {} quarantined, {} aged",
                report.settled, report.promoted, report.quarantined, report.aged
            )?;
        }
        Ok(())
    }

    fn install_embeddings(
        home: &Home,
        out: &mut Output<impl Write>,
        from: Option<PathBuf>,
    ) -> Result<()> {
        let download = home.dir().join("model-download");
        let model_dir = if let Some(from) = from {
            from
        } else {
            fs::create_dir_all(&download)
                .with_context(|| format!("create {}", download.display()))?;
            for file in ModelPack::MODEL_FILES {
                let url = format!(
                    "https://huggingface.co/{}/resolve/main/{file}",
                    ModelPack::MODEL_REPO
                );
                writeln!(out, "Downloading {url}")?;
                let status = Process::new("curl")
                    .args([
                        "--fail",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--output",
                    ])
                    .arg(download.join(file))
                    .arg(&url)
                    .status()
                    .context("run curl")?;
                ensure!(status.success(), "downloading {file} failed");
            }
            download.clone()
        };
        ModelPack::import(&model_dir, &home.embeddings())?;
        if download.exists() {
            fs::remove_dir_all(&download)
                .with_context(|| format!("remove {}", download.display()))?;
        }
        let embedded = Ingest::start(home)?.reembed()?;
        writeln!(
            out,
            "Semantic matching is on. Embedded {embedded} stored revision(s)."
        )?;
        Ok(())
    }

    fn workspace(store: &Store, cwd: Option<PathBuf>) -> Result<Workspace> {
        let cwd = match cwd {
            Some(cwd) => cwd,
            None => env::current_dir().context("find the current directory")?,
        };
        Workspace::resolve(&cwd, store)
    }
}

impl Finding {
    fn embedding_model(home: &Home) -> Self {
        match home.open_embedder() {
            Ok(Some(_)) => Self::Healthy("installed".to_owned()),
            Ok(None) => Self::Note(
                "not installed; semantic matching is off (run `trodden embeddings install` to turn it on)"
                    .to_owned(),
            ),
            Err(error) => Self::Problem(error),
        }
    }
}

impl ExecutableSearch {
    const DEFAULT_EXTENSIONS: &str = ".COM;.EXE;.BAT;.CMD";

    fn from_env() -> Result<Self> {
        let path = env::var_os("PATH").context("PATH is not set")?;
        Ok(Self::new(
            &path,
            env::var_os("PATHEXT").as_deref(),
            cfg!(windows),
        ))
    }

    fn new(path: &OsStr, extensions: Option<&OsStr>, windows: bool) -> Self {
        let extensions = if windows {
            extensions
                .unwrap_or(OsStr::new(Self::DEFAULT_EXTENSIONS))
                .to_string_lossy()
                .split(';')
                .filter(|extension| !extension.is_empty())
                .map(OsString::from)
                .collect()
        } else {
            Vec::new()
        };
        Self {
            dirs: env::split_paths(path).collect(),
            extensions,
        }
    }

    fn find(&self, program: &str) -> Result<PathBuf> {
        self.candidates(program)
            .find(|candidate| candidate.is_file())
            .with_context(|| format!("`{program}` not found"))
    }

    fn candidates<'a>(&'a self, program: &'a str) -> impl Iterator<Item = PathBuf> + 'a {
        self.dirs.iter().flat_map(move |dir| {
            self.extensions
                .iter()
                .map(move |extension| {
                    let mut name = OsString::from(program);
                    name.push(extension);
                    dir.join(name)
                })
                .chain(iter::once(dir.join(program)))
        })
    }
}

impl From<Result<String>> for Finding {
    fn from(result: Result<String>) -> Self {
        match result {
            Ok(detail) => Self::Healthy(detail),
            Err(error) => Self::Problem(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::ErrorKind;

    use super::*;
    use crate::output::Failing;

    struct Scratch {
        home: Home,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = env::temp_dir().join(format!("trodden-cli-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            let home = Home::at(dir);
            home.initialize().expect("home initializes");
            Self { home }
        }

        fn doctor(&self) -> (Result<()>, String) {
            let mut printed = Vec::new();
            let result = Command::doctor(&self.home, &mut Output::new(&mut printed));
            (
                result,
                String::from_utf8(printed).expect("doctor prints UTF-8"),
            )
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.home.dir());
        }
    }

    #[test]
    fn forgetting_an_unknown_procedure_fails() {
        let scratch = Scratch::new("forget-unknown");

        let error = Command::forget(
            &scratch.home,
            &mut Output::new(Vec::new()),
            Some("p_doesnotexist"),
            false,
            false,
        )
        .expect_err("unknown ids are reported");

        assert_eq!(error.to_string(), "no procedure p_doesnotexist");
    }

    #[test]
    fn forgetting_everything_removes_the_hook_log_and_repositories() {
        let scratch = Scratch::new("forget-all");
        let home = &scratch.home;
        fs::write(
            home.hook_log(),
            "2026-10-01T10:00:00Z parse the hook payload\n",
        )
        .expect("hook log is writable");
        home.open_store(Patience::Batch)
            .expect("store opens")
            .remember_repo("/home/dev/shop", "path-5f0c1d2e3a4b6978")
            .expect("repository remembered");

        let mut printed = Vec::new();
        Command::forget(home, &mut Output::new(&mut printed), None, false, true)
            .expect("everything is forgotten");

        assert_eq!(
            String::from_utf8_lossy(&printed),
            "Deleted 0 revision(s).\n"
        );
        assert!(!home.hook_log().exists());
        assert_eq!(
            home.open_store(Patience::Batch)
                .expect("store opens")
                .repo_for_root("/home/dev/shop")
                .expect("repos read"),
            None
        );
        Command::forget(home, &mut Output::new(Vec::new()), None, false, true)
            .expect("forgetting again is harmless");
    }

    #[test]
    fn retiring_waits_for_a_running_ingest() {
        let scratch = Scratch::new("retire-waits");
        let home = scratch.home.clone();
        let procedure = trodden_core::Procedure::example();
        let id = procedure.id.as_str().to_owned();
        home.open_store(Patience::Batch)
            .expect("store opens")
            .upsert(&procedure)
            .expect("procedure stored");
        let ingest = Ingest::start(&home).expect("ingest starts");

        let retiring = std::thread::spawn({
            let home = home.clone();
            let id = id.clone();
            move || Command::retire(&home, &mut Output::new(Vec::new()), &id)
        });
        std::thread::sleep(std::time::Duration::from_millis(300));
        let state_during_ingest = home
            .open_store(Patience::Batch)
            .expect("store opens")
            .revisions(&id)
            .expect("revisions read")[0]
            .procedure
            .state;
        drop(ingest);
        retiring
            .join()
            .expect("retire thread finishes")
            .expect("procedure retires");

        assert_eq!(state_during_ingest, procedure.state);
        assert_eq!(
            home.open_store(Patience::Batch)
                .expect("store opens")
                .revisions(&id)
                .expect("revisions read")[0]
                .procedure
                .state,
            trodden_core::procedure::Lifecycle::Retired
        );
    }

    #[test]
    fn a_missing_embedding_model_is_a_note() {
        let scratch = Scratch::new("doctor-no-model");

        let note = "note     embedding model: not installed; semantic matching is off (run `trodden embeddings install` to turn it on)";

        let (result, printed) = scratch.doctor();

        assert!(printed.lines().any(|line| line == note));
        assert!(!printed.contains("PROBLEM  embedding model"));
        assert_eq!(result.is_ok(), !printed.contains("PROBLEM"));
    }

    #[test]
    fn a_corrupt_embedding_model_is_a_problem() {
        let scratch = Scratch::new("doctor-corrupt-model");
        fs::write(scratch.home.embeddings(), b"TRDEMB\x01\0short").expect("model file is writable");

        let (result, printed) = scratch.doctor();

        assert!(result.is_err());
        assert!(
            printed
                .lines()
                .any(|line| line.starts_with("PROBLEM  embedding model: "))
        );
    }

    #[test]
    fn a_closed_reader_ends_the_command_quietly() {
        let scratch = Scratch::new("closed-reader");
        let mut closed = Output::new(Failing(ErrorKind::BrokenPipe));
        let mut errors = Vec::new();

        let result = Command::status(&scratch.home, &mut closed);
        let code = Cli::exit(result, &mut Output::new(&mut errors));

        assert_eq!(code, ExitCode::SUCCESS);
        assert!(errors.is_empty());
    }

    #[test]
    fn other_failures_are_reported() {
        let mut errors = Vec::new();

        let code = Cli::exit(
            Err(anyhow::anyhow!("no procedure p_doesnotexist")),
            &mut Output::new(&mut errors),
        );

        assert_eq!(code, ExitCode::FAILURE);
        assert_eq!(
            String::from_utf8_lossy(&errors),
            "trodden: no procedure p_doesnotexist\n"
        );
    }

    #[test]
    fn windows_search_tries_each_pathext_extension_before_the_bare_name() {
        let path = env::join_paths(["/opt/node", "/usr/bin"]).expect("dirs join");

        let search = ExecutableSearch::new(&path, Some(OsStr::new(".EXE;;.CMD")), true);

        assert_eq!(
            search.candidates("claude").collect::<Vec<_>>(),
            [
                "/opt/node/claude.EXE",
                "/opt/node/claude.CMD",
                "/opt/node/claude",
                "/usr/bin/claude.EXE",
                "/usr/bin/claude.CMD",
                "/usr/bin/claude",
            ]
            .map(PathBuf::from)
        );
    }

    #[test]
    fn windows_search_falls_back_to_the_default_extensions() {
        let search = ExecutableSearch::new(OsStr::new("/bin"), None, true);

        assert_eq!(
            search.candidates("git").collect::<Vec<_>>(),
            [
                "/bin/git.COM",
                "/bin/git.EXE",
                "/bin/git.BAT",
                "/bin/git.CMD",
                "/bin/git",
            ]
            .map(PathBuf::from)
        );
    }

    #[test]
    fn other_platforms_ignore_pathext() {
        let search = ExecutableSearch::new(OsStr::new("/bin"), Some(OsStr::new(".EXE")), false);

        assert_eq!(
            search.candidates("git").collect::<Vec<_>>(),
            [PathBuf::from("/bin/git")]
        );
    }

    #[test]
    fn windows_search_finds_a_program_installed_with_an_extension() {
        let scratch = Scratch::new("which-extension");
        let installed = scratch.home.dir().join("claude.CMD");
        fs::write(&installed, "").expect("program writes");
        let path = scratch.home.dir().as_os_str();

        let windows = ExecutableSearch::new(path, Some(OsStr::new(".EXE;.CMD")), true);
        let other = ExecutableSearch::new(path, Some(OsStr::new(".EXE;.CMD")), false);

        assert_eq!(windows.find("claude").expect("program is found"), installed);
        assert!(other.find("claude").is_err());
    }
}
