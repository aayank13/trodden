mod checks;
mod lint;
mod task;
mod verify;

use std::{
    fmt::{self, Write as _},
    sync::LazyLock,
};

use jiff::Timestamp;
use regex::Regex;
use sha2::{Digest, Sha256};
use trodden_capture::Command;
use trodden_core::{
    FamilyId, ProcedureId, RepoId, Trace,
    procedure::{
        Condition, Entity, Lifecycle, Outcomes, Procedure, Provenance, Resource, Scope, Slot,
        SlotKind, Source, Step, StepKind, Trigger, Verification as VerifyStep,
    },
    trace::{EventKind, ToolAction, ToolCall, ToolOutcome},
};

use crate::{lint::DangerLint, task::Task, verify::Verification};

pub use checks::{DeclaredCheck, ProjectChecks};

const MAX_STEPS: usize = 25;

const MAX_AVOID: usize = 3;

const MAX_ERRORS: usize = 3;

const TITLE_CHARS: usize = 80;

const COMMAND_NOT_FOUND: i32 = 127;

static SETUP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:npm (?:install|ci|i)|pnpm (?:install|i)|yarn(?: install)?|bun install|pip3? install|python3? -m pip install|uv (?:sync|pip install)|poetry install|bundle install|cargo fetch|go mod download)\b",
    )
    .expect("setup pattern is valid")
});

#[derive(Debug, Clone, PartialEq)]
pub struct Extraction {
    pub first_seq: u32,
    pub last_seq: u32,
    pub started_at: Timestamp,
    pub ended_at: Timestamp,
    pub summary: String,
    pub outcome: TaskOutcome,
    pub tool_calls: u32,
    pub result: Result<Procedure, Rejection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskOutcome {
    Succeeded,
    Failed,
    Unjudged,
}

impl TaskOutcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Unjudged => "unjudged",
        }
    }

    pub fn succeeded(self) -> Option<bool> {
        match self {
            Self::Succeeded => Some(true),
            Self::Failed => Some(false),
            Self::Unjudged => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Rejection {
    NoEdits,
    NotVerified,
    VerificationFailed { command: String },
    TooManySteps { steps: usize },
    ContainsSecret,
    Dangerous { rule: &'static str, command: String },
}

impl fmt::Display for Rejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoEdits => f.write_str("no files were changed"),
            Self::NotVerified => f.write_str("no check ran after the last edit"),
            Self::VerificationFailed { command } => write!(f, "the last check failed: `{command}`"),
            Self::TooManySteps { steps } => write!(f, "{steps} steps is more than {MAX_STEPS}"),
            Self::ContainsSecret => f.write_str("a step depends on a redacted secret"),
            Self::Dangerous { rule, command } => write!(f, "{rule}: `{command}`"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Extractor {
    checks: ProjectChecks,
}

impl Extractor {
    pub fn new(checks: ProjectChecks) -> Self {
        Self { checks }
    }

    pub fn extract(&self, trace: &Trace, repo: &RepoId, session_ended: bool) -> Vec<Extraction> {
        let tasks = Task::split(trace);
        let closed = if session_ended {
            tasks.len()
        } else {
            tasks.len().saturating_sub(1)
        };
        tasks[..closed]
            .iter()
            .map(|task| {
                let builder = Builder::new(trace, repo, &self.checks, task);
                Extraction {
                    first_seq: task.first_seq(),
                    last_seq: task.last_seq(),
                    started_at: task
                        .events
                        .first()
                        .map_or(trace.started_at, |event| event.at),
                    ended_at: task
                        .events
                        .last()
                        .map_or(trace.started_at, |event| event.at),
                    summary: task.summary.to_owned(),
                    outcome: builder.outcome(),
                    tool_calls: u32::try_from(builder.calls.len()).unwrap_or(u32::MAX),
                    result: builder.build(),
                }
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct Call<'a> {
    at: Timestamp,
    call: &'a ToolCall,
}

impl<'a> Call<'a> {
    fn succeeded(&self) -> bool {
        self.call.outcome == ToolOutcome::Succeeded
    }

    fn is_edit(&self) -> bool {
        self.call.action == ToolAction::Edit && self.succeeded() && !self.call.changes.is_empty()
    }

    fn command(&self) -> Option<&'a str> {
        (self.call.action == ToolAction::Run)
            .then_some(self.call.args.command.as_deref())
            .flatten()
    }

    fn exit_code(&self) -> Option<i32> {
        match self.call.outcome {
            ToolOutcome::Failed { exit_code } => exit_code,
            _ => None,
        }
    }
}

struct Builder<'a> {
    trace: &'a Trace,
    repo: &'a RepoId,
    checks: &'a ProjectChecks,
    summary: &'a str,
    calls: Vec<Call<'a>>,
}

impl<'a> Builder<'a> {
    fn new(trace: &'a Trace, repo: &'a RepoId, checks: &'a ProjectChecks, task: &Task<'a>) -> Self {
        let calls = task
            .events
            .iter()
            .filter_map(|event| match &event.kind {
                EventKind::ToolCall(call) if call.outcome != ToolOutcome::Interrupted => {
                    Some(Call { at: event.at, call })
                }
                _ => None,
            })
            .collect();
        Self {
            trace,
            repo,
            checks,
            summary: task.summary,
            calls,
        }
    }

    fn outcome(&self) -> TaskOutcome {
        match self.calls.iter().rposition(Call::is_edit) {
            None => TaskOutcome::Unjudged,
            Some(last_edit) if self.final_verification(last_edit).is_ok() => TaskOutcome::Succeeded,
            Some(_) => TaskOutcome::Failed,
        }
    }

    fn build(&self) -> Result<Procedure, Rejection> {
        let last_edit = self
            .calls
            .iter()
            .rposition(Call::is_edit)
            .ok_or(Rejection::NoEdits)?;
        let verify_at = self.final_verification(last_edit)?;
        let observed = Verification::clean(
            self.calls[verify_at]
                .command()
                .expect("verifications are commands"),
        );

        let mut steps = self.setup_steps();
        steps.extend(self.work_steps(verify_at));
        steps.push(Step {
            kind: StepKind::Verify,
            command: Some(observed.clone()),
            target: None,
            symbols: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
        });
        if steps.len() > MAX_STEPS {
            return Err(Rejection::TooManySteps { steps: steps.len() });
        }
        Self::check_safety(&steps)?;

        let verify = match (self.checks.strongest(), self.checks.find(&observed)) {
            (Some(strongest), found) if found != Some(strongest) => VerifyStep {
                command: strongest.command.clone(),
                expect_exit: 0,
                declared_by: Some(strongest.source.clone()),
            },
            (_, found) => VerifyStep {
                command: observed.clone(),
                expect_exit: 0,
                declared_by: found.map(|check| check.source.clone()),
            },
        };

        Ok(self.procedure(steps, &observed, verify, self.calls[verify_at].at))
    }

    fn final_verification(&self, last_edit: usize) -> Result<usize, Rejection> {
        let learned = self.learned_checks(last_edit);
        let is_check = |call: &Call<'_>| {
            call.command().is_some_and(|command| {
                Verification::is_verify(command)
                    || learned.contains(&command)
                    || self.checks.find(command).is_some()
            })
        };
        let after_edit = || self.calls.iter().enumerate().skip(last_edit + 1);
        let (index, call) = after_edit()
            .rfind(|(_, call)| is_check(call))
            .ok_or(Rejection::NotVerified)?;
        if !call.succeeded() {
            return Err(Rejection::VerificationFailed {
                command: Verification::clean(call.command().unwrap_or_default()),
            });
        }
        let Some(strongest) = self.checks.strongest() else {
            return Ok(index);
        };
        let strongest_run = after_edit().rfind(|(_, call)| {
            call.command()
                .is_some_and(|command| self.checks.find(command) == Some(strongest))
        });
        match strongest_run {
            Some((_, run)) if !run.succeeded() => Err(Rejection::VerificationFailed {
                command: strongest.command.clone(),
            }),
            _ => Ok(index),
        }
    }

    fn learned_checks(&self, last_edit: usize) -> Vec<&'a str> {
        let (before, after) = self.calls.split_at(last_edit + 1);
        before
            .iter()
            .filter(|call| !call.succeeded())
            .filter_map(Call::command)
            .filter(|command| {
                after
                    .iter()
                    .any(|later| later.succeeded() && later.command() == Some(command))
            })
            .collect()
    }

    fn setup_steps(&self) -> Vec<Step> {
        let mut steps: Vec<Step> = Vec::new();
        for command in self
            .calls
            .iter()
            .filter(|call| call.succeeded())
            .filter_map(Call::command)
        {
            if SETUP.is_match(command)
                && !steps
                    .iter()
                    .any(|step| step.command.as_deref() == Some(command))
            {
                steps.push(Step {
                    kind: StepKind::Setup,
                    command: Some(command.to_owned()),
                    target: None,
                    symbols: Vec::new(),
                    reads: Vec::new(),
                    writes: Vec::new(),
                });
            }
        }
        steps
    }

    fn work_steps(&self, verify_at: usize) -> Vec<Step> {
        let mut steps: Vec<Step> = Vec::new();
        let mut phase_start = 0;
        for call in &self.calls[..verify_at] {
            if call.is_edit() {
                for change in &call.call.changes {
                    let existing = steps[phase_start..]
                        .iter_mut()
                        .find(|step| step.target.as_deref() == Some(change.path.as_str()));
                    if let Some(step) = existing {
                        for symbol in &change.symbols {
                            if !step.symbols.contains(symbol) {
                                step.symbols.push(symbol.clone());
                            }
                        }
                        continue;
                    }
                    steps.push(Step {
                        kind: if change.created {
                            StepKind::Create
                        } else {
                            StepKind::Edit
                        },
                        command: None,
                        target: Some(change.path.clone()),
                        symbols: change.symbols.clone(),
                        reads: Vec::new(),
                        writes: vec![Resource::File(change.path.clone())],
                    });
                }
            } else if let Some(command) = call.command().filter(|_| call.succeeded())
                && Self::is_project_task(command)
            {
                if steps.last().and_then(|step| step.command.as_deref()) != Some(command) {
                    steps.push(Step {
                        kind: StepKind::Run,
                        command: Some(command.to_owned()),
                        target: None,
                        symbols: Vec::new(),
                        reads: Vec::new(),
                        writes: Vec::new(),
                    });
                }
                phase_start = steps.len();
            }
        }
        steps
    }

    fn is_project_task(command: &str) -> bool {
        if Verification::is_verify(command) || SETUP.is_match(command) {
            return false;
        }
        let parsed = Command::normalize(command, "");
        let program = parsed.program();
        let mut words = parsed
            .text()
            .split_whitespace()
            .skip_while(|word| word.contains('='));
        let first = words.next().unwrap_or_default();
        let second = words.next().unwrap_or_default();
        let runner = program.split_whitespace().next().unwrap_or_default();
        match program.as_str() {
            "npm run" | "pnpm run" | "yarn run" | "bun run" => true,
            _ if runner == "make" || runner == "just" => true,
            "sh" | "bash" | "zsh" => !second.is_empty() && !second.starts_with('-'),
            "node" | "deno" | "bun" => [".js", ".mjs", ".cjs", ".ts"]
                .iter()
                .any(|extension| second.ends_with(extension)),
            _ if program.starts_with("python ") => !program.starts_with("python -"),
            _ => {
                first.starts_with("./")
                    || first.starts_with("bin/")
                    || first.starts_with("scripts/")
            }
        }
    }

    fn check_safety(steps: &[Step]) -> Result<(), Rejection> {
        for step in steps {
            let texts = step.command.iter().chain(step.target.iter());
            for text in texts {
                if text.contains("[REDACTED:") {
                    return Err(Rejection::ContainsSecret);
                }
                if let Some(rule) = DangerLint::check(text) {
                    return Err(Rejection::Dangerous {
                        rule,
                        command: text.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    fn avoid_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for (index, failed) in self.calls.iter().enumerate() {
            let (Some(command), false) = (failed.command(), failed.succeeded()) else {
                continue;
            };
            let later = &self.calls[index + 1..];
            if later
                .iter()
                .any(|call| call.succeeded() && call.command() == Some(command))
            {
                continue;
            }
            let not_found = failed.exit_code() == Some(COMMAND_NOT_FOUND);
            let replacement = later
                .iter()
                .take_while(|call| not_found || !call.is_edit())
                .find(|call| {
                    call.succeeded()
                        && call.command().is_some_and(|other| {
                            other != command
                                && if not_found {
                                    Self::arguments(other) == Self::arguments(command)
                                } else {
                                    Verification::is_verify(other)
                                        && Verification::is_verify(command)
                                }
                        })
                });
            if let Some(replacement) = replacement.and_then(Call::command) {
                let exit = failed
                    .exit_code()
                    .map_or_else(String::new, |code| format!(" (exit {code})"));
                lines.push(format!("`{command}` failed{exit}; `{replacement}` worked"));
            }
            if lines.len() == MAX_AVOID {
                break;
            }
        }
        lines
    }

    fn arguments(command: &str) -> &str {
        command.split_once(' ').map_or("", |(_, rest)| rest)
    }

    fn procedure(
        &self,
        mut steps: Vec<Step>,
        observed: &str,
        verify: VerifyStep,
        verified_at: Timestamp,
    ) -> Procedure {
        let mut concrete: Vec<&str> = steps
            .iter()
            .filter_map(|step| step.target.as_deref())
            .collect();
        concrete.sort_unstable();
        concrete.dedup();
        let verify_program = Command::normalize(observed, "").program();
        let mut entities: Vec<Entity> = concrete
            .iter()
            .map(|path| Entity::Path((*path).to_owned()))
            .collect();
        for symbol in steps.iter().flat_map(|step| &step.symbols) {
            let entity = Entity::Symbol(symbol.clone());
            if !entities.contains(&entity) {
                entities.push(entity);
            }
        }
        entities.push(Entity::Command(verify_program.clone()));
        let last_edit = self.calls.iter().rposition(Call::is_edit).unwrap_or(0);
        let errors = self.calls[..last_edit]
            .iter()
            .filter(|call| !call.succeeded() && call.exit_code() != Some(COMMAND_NOT_FOUND))
            .filter_map(|call| call.call.error.clone());
        for error in errors {
            let entity = Entity::ErrorSignature(error);
            if !entities.contains(&entity)
                && entities
                    .iter()
                    .filter(|e| matches!(e, Entity::ErrorSignature(_)))
                    .count()
                    < MAX_ERRORS
            {
                entities.push(entity);
            }
        }

        let preconditions = Self::preconditions(&steps, observed, &verify.command);

        let slots = Self::parameterize(&mut steps);

        let mut paths: Vec<&str> = steps
            .iter()
            .filter_map(|step| step.target.as_deref())
            .collect();
        paths.sort_unstable();
        paths.dedup();
        let digest = Sha256::digest(format!(
            "{}\n{}\n{verify_program}",
            self.repo,
            paths.join("\n")
        ));
        let hash = digest.iter().take(6).fold(String::new(), |mut hex, byte| {
            write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
            hex
        });

        Procedure {
            id: ProcedureId::new(format!("p_{hash}")),
            family: FamilyId::new(format!("f_{hash}")),
            revision: 1,
            title: Self::title(self.summary),
            scope: Scope::Repo {
                repo: self.repo.clone(),
            },
            trigger: Trigger {
                entities,
                text: self.summary.to_owned(),
                examples: Vec::new(),
            },
            preconditions,
            steps,
            verify: Some(verify),
            avoid: self.avoid_lines(),
            slots,
            provenance: Provenance {
                sources: vec![Source {
                    session: self.trace.session.clone(),
                    harness: self.trace.harness.clone(),
                    model: self.trace.model.clone(),
                    commit: self.trace.commit.clone(),
                }],
                created_at: verified_at,
                updated_at: verified_at,
            },
            outcomes: Outcomes::default(),
            state: Lifecycle::Candidate,
        }
    }

    fn preconditions(steps: &[Step], observed: &str, verify: &str) -> Vec<Condition> {
        let mut preconditions: Vec<Condition> = Vec::new();
        let mut require = |condition: Condition| {
            if !preconditions.contains(&condition) {
                preconditions.push(condition);
            }
        };
        for step in steps {
            match (step.kind, step.target.as_deref()) {
                (StepKind::Edit, Some(path)) => require(Condition::FileExists {
                    path: path.to_owned(),
                }),
                (StepKind::Create, Some(path)) => {
                    if let Some((dir, _)) = path.rsplit_once('/') {
                        require(Condition::FileExists {
                            path: dir.to_owned(),
                        });
                    }
                }
                _ => {}
            }
        }
        for command in [observed, verify] {
            if let Some(program) = command.split_whitespace().find(|word| !word.contains('=')) {
                require(Condition::ProgramOnPath {
                    program: program.to_owned(),
                });
            }
        }
        preconditions
    }

    fn parameterize(steps: &mut [Step]) -> Vec<Slot> {
        const GENERIC_DIRS: &[&str] = &["", ".", "src", "lib", "app", "pkg", "internal", "source"];
        let mut slots: Vec<Slot> = Vec::new();
        for step in steps
            .iter_mut()
            .filter(|step| step.kind == StepKind::Create)
        {
            let Some(target) = step.target.clone() else {
                continue;
            };
            let (dir, name) = target.rsplit_once('/').unwrap_or(("", &target));
            let (stem, extension) = name.split_once('.').unwrap_or((name, ""));
            if stem.is_empty() {
                continue;
            }
            let parent = dir.rsplit('/').next().unwrap_or(dir);
            let base: String = if GENERIC_DIRS.contains(&parent) {
                "file".to_owned()
            } else {
                let singular = parent
                    .strip_suffix('s')
                    .filter(|rest| rest.len() > 2)
                    .unwrap_or(parent);
                singular
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() {
                            c.to_ascii_lowercase()
                        } else {
                            '_'
                        }
                    })
                    .collect()
            };
            let mut slot_name = base.clone();
            let mut suffix = 2;
            while slots.iter().any(|slot| slot.name == slot_name) {
                slot_name = format!("{base}_{suffix}");
                suffix += 1;
            }
            let file = if extension.is_empty() {
                format!("{{{slot_name}}}")
            } else {
                format!("{{{slot_name}}}.{extension}")
            };
            let template = if dir.is_empty() {
                file
            } else {
                format!("{dir}/{file}")
            };
            step.writes = vec![Resource::File(template.clone())];
            step.target = Some(template);
            slots.push(Slot {
                name: slot_name,
                kind: SlotKind::Identifier,
                examples: vec![stem.to_owned()],
            });
        }
        slots
    }

    fn title(summary: &str) -> String {
        let sentence = summary
            .split(". ")
            .next()
            .unwrap_or(summary)
            .trim()
            .trim_end_matches('.');
        if sentence.chars().count() <= TITLE_CHARS {
            return sentence.to_owned();
        }
        let cut: String = sentence.chars().take(TITLE_CHARS - 3).collect();
        format!("{}...", cut.trim_end())
    }
}

#[cfg(test)]
mod tests {
    use trodden_core::{
        HarnessId, SessionId,
        trace::{Event, FileChange, ToolArgs},
    };

    use super::*;

    struct Sketch(Trace);

    impl Sketch {
        fn new() -> Self {
            Self(Trace {
                session: SessionId::new("0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90"),
                harness: HarnessId::new("claude-code"),
                model: Some("claude-haiku-4-5-20251001".to_owned()),
                cwd: "/work/catalog".to_owned(),
                commit: None,
                started_at: Timestamp::UNIX_EPOCH,
                events: Vec::new(),
            })
        }

        fn push(mut self, kind: EventKind) -> Self {
            let seq = u32::try_from(self.0.events.len()).expect("sketches are small");
            self.0.events.push(Event {
                seq,
                at: Timestamp::UNIX_EPOCH,
                kind,
            });
            self
        }

        fn prompt(self, summary: &str) -> Self {
            self.push(EventKind::Prompt {
                summary: summary.to_owned(),
            })
        }

        fn run(self, command: &str, exit_code: i32) -> Self {
            let outcome = if exit_code == 0 {
                ToolOutcome::Succeeded
            } else {
                ToolOutcome::Failed {
                    exit_code: Some(exit_code),
                }
            };
            self.push(EventKind::ToolCall(ToolCall {
                tool: "Bash".to_owned(),
                action: ToolAction::Run,
                args: ToolArgs {
                    command: Some(command.to_owned()),
                    ..ToolArgs::default()
                },
                outcome,
                changes: Vec::new(),
                duration_ms: None,
                error: None,
            }))
        }

        fn edit(self, path: &str, symbols: &[&str]) -> Self {
            self.push(EventKind::ToolCall(ToolCall {
                tool: "Edit".to_owned(),
                action: ToolAction::Edit,
                args: ToolArgs {
                    path: Some(path.to_owned()),
                    ..ToolArgs::default()
                },
                outcome: ToolOutcome::Succeeded,
                changes: vec![FileChange {
                    path: path.to_owned(),
                    created: false,
                    symbols: symbols.iter().map(|symbol| (*symbol).to_owned()).collect(),
                    lines_added: 1,
                    lines_removed: 1,
                }],
                duration_ms: None,
                error: None,
            }))
        }

        fn create(mut self, path: &str) -> Self {
            self = self.edit(path, &[]);
            if let Some(Event {
                kind: EventKind::ToolCall(call),
                ..
            }) = self.0.events.last_mut()
            {
                call.changes[0].created = true;
            }
            self
        }

        fn extract(&self) -> Vec<Result<Procedure, Rejection>> {
            Extractor::default()
                .extract(&self.0, &RepoId::new("5f0c3a1e"), true)
                .into_iter()
                .map(|extraction| extraction.result)
                .collect()
        }

        fn extract_one(&self) -> Result<Procedure, Rejection> {
            let mut results = self.extract();
            assert_eq!(results.len(), 1, "expected one task");
            results.remove(0)
        }
    }

    fn step_summary(procedure: &Procedure) -> Vec<String> {
        procedure
            .steps
            .iter()
            .map(|step| {
                format!(
                    "{:?} {}",
                    step.kind,
                    step.target
                        .as_deref()
                        .or(step.command.as_deref())
                        .unwrap_or("")
                )
            })
            .collect()
    }

    #[test]
    fn keeps_generators_and_learns_what_to_avoid() {
        let procedure = Sketch::new()
            .prompt("Add a discount_cents field to the Order model")
            .run("python tools/gen_models.py", COMMAND_NOT_FOUND)
            .edit("schema/models.json", &[])
            .run("python3 tools/gen_models.py", 0)
            .run("node --test test/", 1)
            .run("npm test", 0)
            .run("git add -A", 0)
            .extract_one()
            .expect("verified task is admitted");

        assert_eq!(
            step_summary(&procedure),
            [
                "Edit schema/models.json",
                "Run python3 tools/gen_models.py",
                "Verify npm test"
            ]
        );
        assert_eq!(
            procedure.avoid,
            [
                "`python tools/gen_models.py` failed (exit 127); `python3 tools/gen_models.py` worked",
                "`node --test test/` failed (exit 1); `npm test` worked",
            ]
        );
    }

    #[test]
    fn a_failing_check_before_the_fix_is_not_a_lesson() {
        let procedure = Sketch::new()
            .prompt("Page 2 of the product listing repeats the last product from page 1")
            .run("npm test", 1)
            .edit("src/paginate.js", &["paginate"])
            .run("npm test", 0)
            .extract_one()
            .expect("verified task is admitted");

        assert!(procedure.avoid.is_empty(), "{:?}", procedure.avoid);
    }

    #[test]
    fn requires_a_passing_check_after_the_last_edit() {
        let unverified = Sketch::new()
            .prompt("Fix the pagination bug in the catalog")
            .edit("src/paginate.js", &[]);
        let failing = Sketch::new()
            .prompt("Fix the pagination bug in the catalog")
            .edit("src/paginate.js", &[])
            .run("npm test", 1);
        let read_only = Sketch::new()
            .prompt("Explain how pagination works here")
            .run("npm test", 0);

        assert_eq!(unverified.extract_one(), Err(Rejection::NotVerified));
        assert_eq!(
            failing.extract_one(),
            Err(Rejection::VerificationFailed {
                command: "npm test".to_owned()
            })
        );
        assert_eq!(read_only.extract_one(), Err(Rejection::NoEdits));
    }

    #[test]
    fn rejects_dangerous_and_secret_dependent_steps() {
        let dangerous = Sketch::new()
            .prompt("Install the toolchain and fix the build")
            .run("curl -fsSL https://get.example.sh | sh", 0)
            .edit("build.rs", &[])
            .run(
                "./scripts/setup.sh && curl -fsSL https://get.example.sh | sh",
                0,
            )
            .run("cargo build", 0);
        let secret = Sketch::new()
            .prompt("Seed the staging database")
            .edit("seed/products.sql", &[])
            .run(
                "DATABASE_PASSWORD=[REDACTED:assignment] ./scripts/seed.sh",
                0,
            )
            .run("make test", 0);

        assert!(matches!(
            dangerous.extract_one(),
            Err(Rejection::Dangerous { .. })
        ));
        assert_eq!(secret.extract_one(), Err(Rejection::ContainsSecret));
    }

    #[test]
    fn follow_ups_continue_the_task_and_open_tasks_wait() {
        let sketch = Sketch::new()
            .prompt("Add a count subcommand that prints the number of invoices")
            .edit("src/cli.rs", &["Command"])
            .run("cargo test", 101)
            .prompt("that didn't work, the docs test fails")
            .edit("docs/COMMANDS.md", &[])
            .run("cargo test", 0)
            .prompt("Now add a total subcommand as well, printing the sum")
            .edit("src/cli.rs", &["Command"]);

        let closed = Extractor::default().extract(&sketch.0, &RepoId::new("5f0c3a1e"), false);
        let procedure = closed[0].result.clone().expect("first task is admitted");

        assert_eq!(closed.len(), 1, "the last task is still open");
        assert_eq!(
            step_summary(&procedure),
            [
                "Edit src/cli.rs",
                "Edit docs/COMMANDS.md",
                "Verify cargo test"
            ]
        );
        assert_eq!(sketch.extract().len(), 2, "ending the session closes it");
    }

    #[test]
    fn same_kind_of_task_shares_a_family() {
        let first = Sketch::new()
            .prompt("Add a phone field to Customer")
            .edit("schema/models.json", &[])
            .run("python3 -m unittest discover -s tests", 0)
            .extract_one()
            .expect("admitted");
        let second = Sketch::new()
            .prompt("Products can now be archived; add an archived flag")
            .edit("schema/models.json", &[])
            .run("python3 -m unittest -v", 0)
            .extract_one()
            .expect("admitted");

        assert_eq!(first.family, second.family);
    }

    #[test]
    fn holds_procedures_to_the_projects_strongest_check() {
        let checks = ProjectChecks::from_commands([
            ("python3 tools/check.py".to_owned(), "README.md".to_owned()),
            ("python3 -m unittest".to_owned(), "README.md".to_owned()),
        ]);
        let extract = |sketch: Sketch| {
            Extractor::new(checks.clone())
                .extract(&sketch.0, &RepoId::new("5f0c3a1e"), true)
                .remove(0)
        };
        let weaker = Sketch::new()
            .prompt("Add a priority column to tasks")
            .edit("app/models.py", &["Task"])
            .run("python3 -m unittest -v", 0);
        let failed_strongest = Sketch::new()
            .prompt("Add a priority column to tasks")
            .edit("app/models.py", &["Task"])
            .run("python3 tools/check.py", 1)
            .run("python3 -m unittest -v", 0);

        let admitted = extract(weaker);
        let rejected = extract(failed_strongest);

        let procedure = admitted.result.expect("a passing check admits");
        let verify = procedure.verify.clone().expect("procedures have a check");
        assert_eq!(verify.command, "python3 tools/check.py");
        assert_eq!(verify.declared_by.as_deref(), Some("README.md"));
        assert_eq!(
            step_summary(&procedure).last().map(String::as_str),
            Some("Verify python3 -m unittest -v"),
            "the session's own check stays a step"
        );
        assert_eq!(admitted.outcome, TaskOutcome::Succeeded);
        assert_eq!(
            rejected.result,
            Err(Rejection::VerificationFailed {
                command: "python3 tools/check.py".to_owned()
            })
        );
        assert_eq!(rejected.outcome, TaskOutcome::Failed);
    }

    #[test]
    fn created_files_become_slots_shared_by_the_family() {
        let first = Sketch::new()
            .prompt("Add a priority column to tasks")
            .create("migrations/0005_add_priority.sql")
            .edit("app/models.py", &["Task"])
            .run("python3 -m unittest", 0)
            .extract_one()
            .expect("admitted");
        let second = Sketch::new()
            .prompt("Add a due date to tasks")
            .create("migrations/0005_add_due_date.sql")
            .edit("app/models.py", &["Task"])
            .run("python3 -m unittest", 0)
            .extract_one()
            .expect("admitted");

        assert_eq!(
            first.steps[0].target.as_deref(),
            Some("migrations/{migration}.sql")
        );
        assert_eq!(first.slots[0].examples, ["0005_add_priority"]);
        assert_eq!(first.family, second.family);
        assert_eq!(first.steps, second.steps);
        assert!(first.preconditions.contains(&Condition::FileExists {
            path: "migrations".to_owned()
        }));
    }

    #[test]
    fn judges_every_task() {
        let outcomes: Vec<TaskOutcome> = Extractor::default()
            .extract(
                &Sketch::new()
                    .prompt("Explain the pagination module")
                    .run("npm test", 0)
                    .prompt("Fix the off-by-one in src/paginate.js")
                    .edit("src/paginate.js", &[])
                    .run("npm test", 1)
                    .0,
                &RepoId::new("5f0c3a1e"),
                true,
            )
            .into_iter()
            .map(|extraction| extraction.outcome)
            .collect();

        assert_eq!(outcomes, [TaskOutcome::Unjudged, TaskOutcome::Failed]);
    }

    #[test]
    fn titles_are_first_sentences() {
        assert_eq!(Builder::title("Fix it. Then run the tests."), "Fix it");
        assert_eq!(
            Builder::title(&"word ".repeat(40)).chars().count(),
            TITLE_CHARS
        );
    }
}
