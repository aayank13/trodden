use std::{
    env, fs,
    path::PathBuf,
    process::{Command as Process, ExitCode},
};

use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, Subcommand};
use trodden::{Home, Ingest, IngestReport, Workspace};
use trodden_core::RepoId;
use trodden_embed::ModelPack;
use trodden_recall::{Abstention, Decision, Envelope, Match, Outcome, Query, Recall};
use trodden_store::{Forget, Patience, Store};

use crate::mcp::Server;

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
        #[arg(help = "Harness name: `claude-code`")]
        harness: String,
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

impl Cli {
    pub(crate) fn run() -> ExitCode {
        let cli = Self::parse();
        match cli.command.execute() {
            Ok(()) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("trodden: {error:#}");
                ExitCode::FAILURE
            }
        }
    }
}

impl Command {
    fn execute(self) -> Result<()> {
        let home = Home::locate()?;
        if !matches!(self, Self::Init | Self::Doctor | Self::Hook { .. }) {
            ensure!(home.is_initialized(), "run `trodden init` first");
        }
        match self {
            Self::Init => Self::init(&home),
            Self::Status => Self::status(&home),
            Self::Doctor => Self::doctor(&home),
            Self::List {
                all_repos,
                revisions,
            } => Self::list(&home, all_repos, revisions),
            Self::Show { id } => Self::show(&home, &id),
            Self::Recall {
                prompt,
                cwd,
                explain,
            } => Self::recall(&home, &prompt, cwd, explain),
            Self::Rejected { limit } => Self::rejected(&home, limit),
            Self::Outcomes => Self::outcomes(&home),
            Self::Retire { id } => Self::retire(&home, &id),
            Self::Config { setting } => Self::config(&home, setting),
            Self::Pause => home.open_store(Patience::Batch)?.set_paused(true),
            Self::Resume => home.open_store(Patience::Batch)?.set_paused(false),
            Self::Forget { id, repo, all, .. } => Self::forget(&home, id.as_deref(), repo, all),
            Self::Backfill { projects } => Self::backfill(&home, projects),
            Self::Ingest {
                transcript,
                ended,
                quiet,
            } => {
                let report = Ingest::start(&home)?.claude_code(&transcript, ended)?;
                if !quiet {
                    Self::print_report(&report);
                }
                Ok(())
            }
            Self::Embeddings {
                command: EmbeddingsCommand::Install { from },
            } => Self::install_embeddings(&home, from),
            Self::Mcp => Server::serve(home),
            Self::Hook { .. } => {
                bail!("hooks read their payload from stdin; see `trodden help hook`")
            }
        }
    }

    fn init(home: &Home) -> Result<()> {
        let fresh = !home.is_initialized();
        home.initialize()?;
        println!("Data directory: {}", home.dir().display());
        if fresh {
            println!(
                "\nTrodden now learns from Claude Code sessions in this account. It stores\n\
                 the first line of each prompt (up to 200 characters), commands, file paths,\n\
                 touched function names, exit codes and one normalized line per error, with\n\
                 secrets redacted. It never stores file contents, other tool output or the\n\
                 rest of a prompt, and makes no network calls. `trodden pause` stops it;\n\
                 `trodden forget` deletes."
            );
        }
        println!("\nNext steps:");
        println!("  1. Install the Claude Code plugin:");
        println!("       claude plugin marketplace add aayank13/trodden");
        println!("       claude plugin install trodden@trodden");
        if home.embedder().is_none() {
            println!("  2. Optional: enable semantic matching (one-time 32 MB download):");
            println!("       trodden embeddings install");
        }
        println!(
            "  {}. Optional: learn from past sessions:",
            if home.embedder().is_none() { 3 } else { 2 }
        );
        println!("       trodden backfill");
        Ok(())
    }

    fn status(home: &Home) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let stats = store.stats()?;
        println!("Data directory     {}", home.dir().display());
        println!(
            "Capture and recall {}",
            if store.paused()? { "paused" } else { "on" }
        );
        println!(
            "Semantic matching  {}",
            if home.semantic().is_some() {
                "on"
            } else {
                "off (run `trodden embeddings install`)"
            }
        );
        println!(
            "Procedures         {} ({} revisions)",
            stats.procedures, stats.revisions
        );
        println!("Sessions ingested  {}", stats.sessions);
        println!("Rejected tasks     {}", stats.rejections);
        println!("Injections         {}", stats.injections);
        println!(
            "Holdout            {}",
            store
                .holdout_rate()?
                .map_or_else(|| "auto".to_owned(), |rate| format!("{:.0}%", rate * 100.0))
        );
        Ok(())
    }

    fn doctor(home: &Home) -> Result<()> {
        let mut problems = 0;
        let mut check = |label: &str, result: Result<String>| match result {
            Ok(detail) => println!("ok       {label}: {detail}"),
            Err(error) => {
                problems += 1;
                println!("PROBLEM  {label}: {error:#}");
            }
        };
        check(
            "data directory",
            if home.is_initialized() {
                Ok(home.dir().display().to_string())
            } else {
                Err(anyhow::anyhow!("not initialized; run `trodden init`"))
            },
        );
        if home.is_initialized() {
            check(
                "database",
                home.open_store(Patience::Batch)
                    .and_then(|store| store.stats())
                    .map(|stats| format!("{} procedures", stats.procedures)),
            );
        }
        check(
            "embedding model",
            home.embedder()
                .map(|_| "installed".to_owned())
                .context("not installed; semantic matching is off"),
        );
        for program in ["trodden", "git", "claude"] {
            check(
                &format!("`{program}` on PATH"),
                Self::which(program).map(|path| path.display().to_string()),
            );
        }
        if let Ok(log) = fs::read_to_string(home.hook_log()) {
            for line in log.lines().rev().take(3) {
                println!("note     recent hook error: {line}");
            }
        }
        ensure!(problems == 0, "{problems} problem(s) found");
        Ok(())
    }

    fn list(home: &Home, all_repos: bool, revisions: bool) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let repo = if all_repos {
            None
        } else {
            Some(Self::workspace(&store, None)?.repo)
        };
        let rows = store.list(repo.as_ref().map(RepoId::as_str), revisions)?;
        if rows.is_empty() {
            println!(
                "No procedures yet{}.",
                if all_repos { "" } else { " in this repository" }
            );
            return Ok(());
        }
        println!(
            "{:<16} {:>3}  {:<11} {:>8} {:>7}  TITLE",
            "ID", "REV", "STATE", "SESSIONS", "WORKED"
        );
        for row in rows {
            let procedure = row.procedure;
            let state = serde_json::to_value(procedure.state).context("name a lifecycle state")?;
            let judged = procedure.outcomes.successes + procedure.outcomes.failures;
            let worked = if judged == 0 {
                "-".to_owned()
            } else {
                format!("{}/{judged}", procedure.outcomes.successes)
            };
            println!(
                "{:<16} {:>3}  {:<11} {:>8} {:>7}  {}",
                procedure.id,
                procedure.revision,
                state.as_str().unwrap_or_default(),
                procedure.provenance.sources.len(),
                worked,
                procedure.title
            );
        }
        Ok(())
    }

    fn show(home: &Home, id: &str) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let revisions = store.revisions(id)?;
        ensure!(!revisions.is_empty(), "no procedure {id}");
        for row in revisions {
            println!("{}\n", Envelope::render(&row.procedure));
            println!(
                "{}\n",
                serde_json::to_string_pretty(&row.procedure).context("serialize a procedure")?
            );
        }
        Ok(())
    }

    fn recall(home: &Home, prompt: &str, cwd: Option<PathBuf>, explain: bool) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let workspace = Self::workspace(&store, cwd)?;
        let mut recall = Recall::new(&store, home.semantic());
        let outcome = recall.recall(&Query {
            prompt,
            repo: workspace.repo.as_str(),
            root: &workspace.root,
            session: None,
        })?;
        if explain {
            Self::print_explanation(&outcome);
        }
        match outcome.decision {
            Decision::Inject(chosen) | Decision::Withhold(chosen) => {
                println!("{}", Envelope::render(&chosen.row.procedure));
            }
            Decision::Abstain(reason) => {
                println!("Nothing to inject: {}.", Self::describe(&reason));
            }
        }
        Ok(())
    }

    fn print_explanation(outcome: &Outcome) {
        if outcome.candidates.is_empty() {
            return;
        }
        println!(
            "{:<16} {:>7} {:>9} {:>9}  {:<24} TITLE",
            "ID", "FUSED", "LEXICAL", "COSINE", "EXACT"
        );
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
            println!(
                "{:<16} {:>7.4} {:>9} {:>9}  {:<24} {}",
                row.procedure.id,
                signals.fused,
                lexical,
                cosine,
                exact.join(","),
                row.procedure.title
            );
        }
        println!();
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

    fn outcomes(home: &Home) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let summaries = store.outcome_summaries()?;
        if summaries.is_empty() {
            println!("No procedure has been recalled yet.");
            return Ok(());
        }
        println!(
            "{:<16} {:>10} {:>10} {:>14}  TITLE",
            "ID", "INJECTED", "HELD OUT", "TOOL CALLS"
        );
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
            println!(
                "{:<16} {:>10} {:>10} {:>14}  {}",
                summary.procedure,
                rate(summary.injected),
                rate(summary.held_out),
                calls,
                summary.title
            );
        }
        println!(
            "\nINJECTED and HELD OUT count tasks that ended with a passing check, out of those\n\
             that changed files. TOOL CALLS compares the mean effort of those tasks."
        );
        Ok(())
    }

    fn retire(home: &Home, id: &str) -> Result<()> {
        let retired = home.open_store(Patience::Batch)?.retire(id)?;
        ensure!(retired > 0, "no procedure {id}");
        Ingest::start(home)?.rebuild_index()?;
        println!("Retired {retired} revision(s) of {id}.");
        Ok(())
    }

    fn config(home: &Home, setting: Option<Setting>) -> Result<()> {
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
        println!("holdout          {holdout}");
        println!(
            "verify-reminder  {}",
            if store.verify_reminder()? {
                "on"
            } else {
                "off"
            }
        );
        Ok(())
    }

    fn rejected(home: &Home, limit: usize) -> Result<()> {
        let store = home.open_store(Patience::Batch)?;
        let records = store.rejections(limit)?;
        if records.is_empty() {
            println!("No rejected tasks.");
        }
        for record in records {
            println!(
                "{}  {}\n    {}",
                record.at,
                record.summary,
                record.rejection.unwrap_or_default()
            );
        }
        Ok(())
    }

    fn forget(home: &Home, id: Option<&str>, repo: bool, all: bool) -> Result<()> {
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
        Ingest::start(home)?.rebuild_index()?;
        println!("Deleted {deleted} revision(s).");
        Ok(())
    }

    fn backfill(home: &Home, projects: Option<PathBuf>) -> Result<()> {
        let projects = match projects {
            Some(projects) => projects,
            None => PathBuf::from(env::var_os("HOME").context("find the home directory")?)
                .join(".claude/projects"),
        };
        let (report, failures) = Ingest::start(home)?.backfill_claude_code(&projects)?;
        Self::print_report(&report);
        for (transcript, error) in &failures {
            eprintln!("skipped {}: {error}", transcript.display());
        }
        Ok(())
    }

    fn print_report(report: &IngestReport) {
        println!(
            "{} session(s), {} task(s): {} new procedure(s), {} confirmed, {} new revision(s), {} rejected",
            report.sessions,
            report.tasks,
            report.created,
            report.refreshed,
            report.revised,
            report.rejected.values().sum::<usize>()
        );
        for (reason, count) in &report.rejected {
            println!("  {count:>4}  {reason}");
        }
        if report.settled + report.promoted + report.quarantined + report.aged > 0 {
            println!(
                "{} injection(s) judged; {} revision(s) promoted, {} quarantined, {} aged",
                report.settled, report.promoted, report.quarantined, report.aged
            );
        }
    }

    fn install_embeddings(home: &Home, from: Option<PathBuf>) -> Result<()> {
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
                println!("Downloading {url}");
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
        println!("Semantic matching is on. Embedded {embedded} stored revision(s).");
        Ok(())
    }

    fn workspace(store: &Store, cwd: Option<PathBuf>) -> Result<Workspace> {
        let cwd = match cwd {
            Some(cwd) => cwd,
            None => env::current_dir().context("find the current directory")?,
        };
        Workspace::resolve(&cwd, store)
    }

    fn which(program: &str) -> Result<PathBuf> {
        let path = env::var_os("PATH").context("PATH is not set")?;
        env::split_paths(&path)
            .map(|dir| dir.join(program))
            .find(|candidate| candidate.is_file())
            .with_context(|| format!("`{program}` not found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(self.home.dir());
        }
    }

    #[test]
    fn forgetting_an_unknown_procedure_fails() {
        let scratch = Scratch::new("forget-unknown");

        let error = Command::forget(&scratch.home, Some("p_doesnotexist"), false, false)
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

        Command::forget(home, None, false, true).expect("everything is forgotten");

        assert!(!home.hook_log().exists());
        assert_eq!(
            home.open_store(Patience::Batch)
                .expect("store opens")
                .repo_for_root("/home/dev/shop")
                .expect("repos read"),
            None
        );
        Command::forget(home, None, false, true).expect("forgetting again is harmless");
    }
}
