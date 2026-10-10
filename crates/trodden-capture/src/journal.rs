use anyhow::{Context, Result};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use trodden_core::{
    Trace,
    trace::{EventKind, FileChange, ToolAction, ToolArgs, ToolCall, ToolOutcome},
};
use trodden_redact::Redactor;

use crate::{
    ErrorSignature,
    builder::{Prompts, TraceBuilder},
    command::Command,
    diff::Diff,
    evidence::CheckOutput,
    symbols::SymbolFinder,
};

// Hooks record what some agents never write to disk. Lines are redacted and reduced when
// written; paths become relative only when the whole journal is read.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Observation {
    pub at: Timestamp,
    pub session: String,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(flatten)]
    pub observed: Observed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Observed {
    Prompt {
        text: String,
    },
    Call {
        tool: String,
        action: ToolAction,
        #[serde(default, skip_serializing_if = "ToolArgs::is_empty")]
        args: ToolArgs,
        outcome: ToolOutcome,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        changes: Vec<FileChange>,
    },
    Compaction {
        automatic: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ran<'a> {
    pub output: &'a str,
    pub exit_code: Option<i32>,
    pub failed: bool,
    pub interrupted: bool,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edit<'a> {
    Replaced { old: &'a str, new: &'a str },
    Created { content: &'a str },
    Patched { diff: &'a str },
    Unknown,
}

#[derive(Debug)]
pub struct Observer<'a> {
    pub session: &'a str,
    pub cwd: &'a str,
    pub at: Timestamp,
    pub model: Option<&'a str>,
    pub redactor: &'a Redactor,
}

impl Observer<'_> {
    fn observation(&self, observed: Observed) -> Observation {
        Observation {
            at: self.at,
            session: self.session.to_owned(),
            cwd: self.cwd.to_owned(),
            model: self.model.map(str::to_owned),
            observed,
        }
    }

    fn call(&self, tool: &str, action: ToolAction, args: ToolArgs) -> Observed {
        Observed::Call {
            tool: tool.to_owned(),
            action,
            args,
            outcome: ToolOutcome::Succeeded,
            error: None,
            duration_ms: None,
            changes: Vec::new(),
        }
    }

    pub fn prompt(&self, text: &str) -> Observation {
        self.observation(Observed::Prompt {
            text: TraceBuilder::summarize(text, &Journal::PROMPTS, self.redactor)
                .unwrap_or_default(),
        })
    }

    pub fn command(&self, tool: &str, command: &str, ran: Ran<'_>) -> Observation {
        let verdict = CheckOutput::verdict(command, ran.output);
        let outcome = match (ran.interrupted, ran.failed, verdict) {
            (true, _, _) => ToolOutcome::Interrupted,
            (false, true, Some(true)) | (false, false, None | Some(true)) => ToolOutcome::Succeeded,
            (false, true, _) => ToolOutcome::Failed {
                exit_code: ran.exit_code,
            },
            (false, false, Some(false)) => ToolOutcome::Failed { exit_code: None },
        };
        let error = matches!(outcome, ToolOutcome::Failed { .. })
            .then(|| ErrorSignature::of(ran.output, self.redactor))
            .flatten();
        // Normalized here so heredoc bodies never reach the disk.
        let normalized = Command::normalize(command, self.cwd);
        let args = ToolArgs {
            command: Some(self.redactor.redact_secrets(normalized.text()).into_owned()),
            ..ToolArgs::default()
        };
        self.observation(Observed::Call {
            tool: tool.to_owned(),
            action: normalized.action(),
            args,
            outcome,
            error,
            duration_ms: ran.duration_ms,
            changes: Vec::new(),
        })
    }

    pub fn edit(&self, tool: &str, path: &str, edit: Edit<'_>, succeeded: bool) -> Observation {
        let (diff, created) = match edit {
            Edit::Replaced { old, new } => (Diff::replaced(old, new), false),
            Edit::Created { content } => (Diff::created(content), true),
            Edit::Patched { diff } => (Diff::unified(diff), false),
            Edit::Unknown => (Diff::default(), false),
        };
        let path = self.redactor.redact_secrets(path).into_owned();
        let changes = if succeeded {
            vec![FileChange {
                path: path.clone(),
                created,
                symbols: SymbolFinder::find(diff.changed.iter().map(String::as_str), None, &[]),
                lines_added: diff.added,
                lines_removed: diff.removed,
            }]
        } else {
            Vec::new()
        };
        self.observation(Observed::Call {
            tool: tool.to_owned(),
            action: ToolAction::Edit,
            args: ToolArgs {
                path: Some(path),
                ..ToolArgs::default()
            },
            outcome: if succeeded {
                ToolOutcome::Succeeded
            } else {
                ToolOutcome::Failed { exit_code: None }
            },
            error: None,
            duration_ms: None,
            changes,
        })
    }

    pub fn read(&self, tool: &str, path: &str) -> Observation {
        let args = ToolArgs {
            path: Some(self.redactor.redact_secrets(path).into_owned()),
            ..ToolArgs::default()
        };
        self.observation(self.call(tool, ToolAction::Read, args))
    }

    pub fn search(&self, tool: &str, pattern: Option<&str>, path: Option<&str>) -> Observation {
        let args = ToolArgs {
            pattern: pattern.map(|pattern| self.redactor.redact_secrets(pattern).into_owned()),
            path: path.map(|path| self.redactor.redact_secrets(path).into_owned()),
            ..ToolArgs::default()
        };
        self.observation(self.call(tool, ToolAction::Search, args))
    }

    pub fn fetch(&self, tool: &str, url: Option<&str>, query: Option<&str>) -> Observation {
        let args = ToolArgs {
            url: url.map(|url| self.redactor.redact_secrets(url).into_owned()),
            query: query.map(|query| self.redactor.redact_secrets(query).into_owned()),
            ..ToolArgs::default()
        };
        self.observation(self.call(tool, ToolAction::Fetch, args))
    }

    pub fn other(&self, tool: &str, action: ToolAction) -> Observation {
        self.observation(self.call(tool, action, ToolArgs::default()))
    }

    pub fn compaction(&self, automatic: bool) -> Observation {
        self.observation(Observed::Compaction { automatic })
    }
}

impl Observation {
    pub fn to_line(&self) -> Result<String> {
        let mut line = serde_json::to_string(self).context("encode a journal line")?;
        line.push('\n');
        Ok(line)
    }
}

#[derive(Debug)]
pub struct Journal;

impl Journal {
    const PROMPTS: Prompts = Prompts {
        skipped: &[],
        pasted: None,
    };

    pub fn parse(text: &str, harness: &'static str, redactor: &Redactor) -> Result<Trace> {
        let mut builder = TraceBuilder::new(harness, redactor);
        for observation in text
            .lines()
            .filter_map(|line| serde_json::from_str::<Observation>(line).ok())
        {
            builder.session(&observation.session);
            builder.directory(&observation.cwd);
            builder.time_at(observation.at);
            if let Some(model) = &observation.model {
                builder.model(model);
            }
            match observation.observed {
                Observed::Prompt { text } => builder.prompt(&text, &Self::PROMPTS),
                Observed::Compaction { automatic } => builder.compaction(automatic),
                Observed::Call {
                    tool,
                    action,
                    args,
                    outcome,
                    error,
                    duration_ms,
                    changes,
                } => {
                    let call = Self::relative(
                        &builder,
                        ToolCall {
                            tool,
                            action,
                            args,
                            outcome,
                            changes,
                            duration_ms,
                            error,
                        },
                    );
                    builder.emit(EventKind::ToolCall(call));
                }
            }
        }
        builder.finish("no session found in journal")
    }

    pub fn working_directory(text: &str) -> Option<String> {
        text.lines()
            .filter_map(|line| serde_json::from_str::<Observation>(line).ok())
            .map(|observation| observation.cwd)
            .find(|cwd| !cwd.is_empty())
    }

    fn relative(builder: &TraceBuilder<'_>, mut call: ToolCall) -> ToolCall {
        if let Some(command) = &call.args.command {
            call.args.command = builder.shell(command).0.command;
        }
        if let Some(path) = &call.args.path {
            call.args.path = Some(builder.relative(path));
        }
        if let Some(pattern) = &call.args.pattern {
            call.args.pattern = Some(builder.search_pattern(pattern, true));
        }
        for change in &mut call.changes {
            change.path = builder.relative(&change.path);
        }
        call
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Session {
        redactor: Redactor,
        lines: String,
    }

    impl Session {
        fn new() -> Self {
            Self {
                redactor: Redactor::with_home("/home/dev"),
                lines: String::new(),
            }
        }

        fn observe(&mut self, cwd: &str, at: &str, record: impl Fn(&Observer<'_>) -> Observation) {
            let observer = Observer {
                session: "conv-1",
                cwd,
                at: at.parse().expect("valid timestamp"),
                model: Some("gpt-5.5"),
                redactor: &self.redactor,
            };
            self.lines
                .push_str(&record(&observer).to_line().expect("line encodes"));
        }

        fn trace(&self) -> Trace {
            Journal::parse(&self.lines, "cursor", &self.redactor).expect("journal parses")
        }

        fn calls(&self) -> Vec<ToolCall> {
            self.trace()
                .events
                .into_iter()
                .filter_map(|event| match event.kind {
                    EventKind::ToolCall(call) => Some(call),
                    _ => None,
                })
                .collect()
        }
    }

    #[test]
    fn a_journal_reads_back_as_a_trace() {
        let mut session = Session::new();
        session.observe("/home/dev/shop", "2026-10-09T10:00:00Z", |o| {
            o.prompt("Page 2 repeats the last product. Fix it.\nMore detail here")
        });
        session.observe("/home/dev/shop", "2026-10-09T10:00:05Z", |o| {
            o.edit(
                "StrReplace",
                "/home/dev/shop/src/paginate.js",
                Edit::Replaced {
                    old: "export function paginate(items) {\n  return items.slice(start, end + 1);\n}",
                    new: "export function paginate(items) {\n  return items.slice(start, end);\n}",
                },
                true,
            )
        });
        session.observe("/home/dev/shop", "2026-10-09T10:00:09Z", |o| {
            o.command(
                "Shell",
                "npm test",
                Ran {
                    output: "1 passing",
                    exit_code: Some(0),
                    failed: false,
                    interrupted: false,
                    duration_ms: Some(900),
                },
            )
        });

        let trace = session.trace();

        assert_eq!(trace.session.as_str(), "conv-1");
        assert_eq!(trace.harness.as_str(), "cursor");
        assert_eq!(trace.cwd, "~/shop");
        assert_eq!(trace.model.as_deref(), Some("gpt-5.5"));
        assert!(matches!(
            &trace.events[0].kind,
            EventKind::Prompt { summary } if summary == "Page 2 repeats the last product. Fix it."
        ));
        let calls = session.calls();
        assert_eq!(calls[0].changes[0].path, "src/paginate.js");
        assert_eq!(
            (
                calls[0].changes[0].lines_added,
                calls[0].changes[0].lines_removed
            ),
            (1, 1)
        );
        assert_eq!(calls[1].args.command.as_deref(), Some("npm test"));
        assert_eq!(calls[1].action, ToolAction::Run);
        assert_eq!(calls[1].outcome, ToolOutcome::Succeeded);
        assert_eq!(calls[1].duration_ms, Some(900));
    }

    #[test]
    fn failed_commands_keep_their_exit_code_and_error() {
        let mut session = Session::new();
        session.observe("/home/dev/shop", "2026-10-09T10:00:09Z", |o| {
            o.command(
                "Shell",
                "npm test",
                Ran {
                    output: "Error: Cannot find module 'left-pad'\n",
                    exit_code: Some(1),
                    failed: true,
                    interrupted: false,
                    duration_ms: None,
                },
            )
        });

        let call = session.calls().remove(0);

        assert_eq!(call.outcome, ToolOutcome::Failed { exit_code: Some(1) });
        assert!(call.error.is_some());
    }

    #[test]
    fn commands_from_a_subdirectory_replay_from_the_first_one() {
        let mut session = Session::new();
        session.observe("/home/dev/shop", "2026-10-09T10:00:00Z", |o| {
            o.prompt("Fix the web tests")
        });
        session.observe("/home/dev/shop/web", "2026-10-09T10:00:09Z", |o| {
            o.command(
                "Shell",
                "npm test",
                Ran {
                    output: "",
                    exit_code: Some(0),
                    failed: false,
                    interrupted: false,
                    duration_ms: None,
                },
            )
        });

        assert_eq!(
            session.calls()[0].args.command.as_deref(),
            Some("cd web && npm test")
        );
    }

    #[test]
    fn secrets_never_reach_the_journal() {
        let redactor = Redactor::with_home("/home/dev");
        let observer = Observer {
            session: "conv-1",
            cwd: "/home/dev/shop",
            at: Timestamp::UNIX_EPOCH,
            model: None,
            redactor: &redactor,
        };
        let key = ["sk", "ant", "api03", "Ab3Zq8Lm3KpQ7vX2nB9wR4tY6uI1oP5aS0dF"].join("-");

        let line = observer
            .command(
                "Shell",
                &format!("ANTHROPIC_API_KEY={key} npm test"),
                Ran {
                    output: "",
                    exit_code: Some(0),
                    failed: false,
                    interrupted: false,
                    duration_ms: None,
                },
            )
            .to_line()
            .expect("line encodes");

        assert!(!line.contains(&key), "{line}");
    }

    #[test]
    fn heredoc_bodies_never_reach_the_journal() {
        let redactor = Redactor::with_home("/home/dev");
        let observer = Observer {
            session: "conv-1",
            cwd: "/home/dev/shop",
            at: Timestamp::UNIX_EPOCH,
            model: None,
            redactor: &redactor,
        };

        let line = observer
            .command(
                "Shell",
                "cat > config.py <<'EOF'\nDB_HOST = 'prod-db.internal'\nEOF\nnpm test",
                Ran {
                    output: "",
                    exit_code: Some(0),
                    failed: false,
                    interrupted: false,
                    duration_ms: None,
                },
            )
            .to_line()
            .expect("line encodes");

        assert!(!line.contains("prod-db.internal"), "{line}");
    }

    #[test]
    fn journals_keep_only_the_prompt_summary() {
        let redactor = Redactor::with_home("/home/dev");
        let observer = Observer {
            session: "conv-1",
            cwd: "/home/dev/shop",
            at: Timestamp::UNIX_EPOCH,
            model: None,
            redactor: &redactor,
        };
        let prompt = format!("{}\nThe private details of the task", "x".repeat(300));

        let Observed::Prompt { text } = observer.prompt(&prompt).observed else {
            panic!("a prompt observation");
        };

        assert_eq!(text, "x".repeat(TraceBuilder::SUMMARY_CHARS));
    }

    #[test]
    fn unreadable_lines_are_skipped() {
        let mut session = Session::new();
        session.observe("/home/dev/shop", "2026-10-09T10:00:00Z", |o| {
            o.prompt("Fix the build")
        });
        session.lines.push_str("{\"type\":\"call\",\"tru\n");

        assert_eq!(session.trace().events.len(), 1);
    }
}
