use anyhow::{Context, Result};
use jiff::Timestamp;
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use trodden_core::{Procedure, procedure::Lifecycle};
use trodden_learn::Evidence;

use crate::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Cue {
    Prompt,
    Failure,
}

impl Cue {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Failure => "failure",
        }
    }

    fn parse(text: &str) -> Self {
        if text == "failure" {
            Self::Failure
        } else {
            Self::Prompt
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Injection {
    pub session: String,
    pub procedure: String,
    pub revision: u32,
    pub holdout: bool,
    pub cue: Cue,
    pub at: Timestamp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InjectionRecord {
    pub rowid: i64,
    pub injection: Injection,
    pub task: Option<u32>,
    pub outcome: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FamilyEvidence {
    pub revisions: Vec<(u32, Lifecycle, Evidence)>,
    pub holdout: Evidence,
}

impl FamilyEvidence {
    pub fn injected(&self) -> Evidence {
        self.revisions
            .iter()
            .fold(Evidence::default(), |sum, (_, _, evidence)| {
                Evidence::new(
                    sum.successes + evidence.successes,
                    sum.failures + evidence.failures,
                )
            })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub procedure: String,
    pub revision: u32,
    pub state: Lifecycle,
    pub used_at: Timestamp,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OutcomeSummary {
    pub procedure: String,
    pub title: String,
    pub injected: Evidence,
    pub held_out: Evidence,
    pub injected_tool_calls: Option<f64>,
    pub held_out_tool_calls: Option<f64>,
}

impl Store {
    pub fn record_injection(&self, injection: &Injection) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO injections (session, procedure, revision, holdout, cue, at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    injection.session,
                    injection.procedure,
                    injection.revision,
                    injection.holdout,
                    injection.cue.as_str(),
                    injection.at.to_string()
                ],
            )
            .context("record an injection")?;
        Ok(())
    }

    pub fn was_injected(&self, session: &str, procedure: &str) -> Result<bool> {
        self.conn
            .query_row(
                "SELECT EXISTS (SELECT 1 FROM injections WHERE session = ?1 AND procedure = ?2)",
                params![session, procedure],
                |row| row.get(0),
            )
            .context("check earlier injections")
    }

    pub fn injections(&self, session: Option<&str>) -> Result<Vec<InjectionRecord>> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT rowid, session, procedure, revision, holdout, cue, at, task, outcome
                 FROM injections WHERE ?1 IS NULL OR session = ?1 ORDER BY rowid",
            )
            .context("prepare the injection listing")?;
        let rows = statement
            .query_map([session], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, u32>(3)?,
                    row.get::<_, bool>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<u32>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            })
            .context("list injections")?;
        rows.map(|row| {
            let (rowid, session, procedure, revision, holdout, cue, at, task, outcome) =
                row.context("read an injection")?;
            Ok(InjectionRecord {
                rowid,
                injection: Injection {
                    session,
                    procedure,
                    revision,
                    holdout,
                    cue: Cue::parse(&cue),
                    at: at.parse().context("parse an injection time")?,
                },
                task,
                outcome,
            })
        })
        .collect()
    }

    pub fn settle_injection(
        &self,
        record: &InjectionRecord,
        task: u32,
        outcome: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "UPDATE injections SET task = ?1, outcome = ?2 WHERE rowid = ?3",
                params![task, outcome, record.rowid],
            )
            .context("settle an injection")?;
        self.mark_used(
            &record.injection.procedure,
            record.injection.revision,
            record.injection.at,
        )
    }

    pub fn mark_used(&self, procedure: &str, revision: u32, at: Timestamp) -> Result<()> {
        self.conn
            .execute(
                "UPDATE procedures SET used_at = max(coalesce(used_at, ''), ?1)
                 WHERE id = ?2 AND revision = ?3",
                params![at.to_string(), procedure, revision],
            )
            .context("mark a revision as used")?;
        Ok(())
    }

    pub fn family_evidence(&self, procedure: &str) -> Result<FamilyEvidence> {
        let mut evidence = FamilyEvidence::default();
        let mut revisions = self
            .conn
            .prepare_cached(
                "SELECT revision, state FROM procedures WHERE id = ?1 ORDER BY revision",
            )
            .context("prepare the revision listing")?;
        for row in revisions
            .query_map([procedure], |row| {
                Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?))
            })
            .context("list revisions")?
        {
            let (revision, state) = row.context("read a revision")?;
            evidence
                .revisions
                .push((revision, Self::parse_state(&state), Evidence::default()));
        }

        let mut counts = self
            .conn
            .prepare_cached(
                "SELECT revision, holdout, outcome = 'succeeded', COUNT(*) FROM injections
                 WHERE procedure = ?1 AND outcome IN ('succeeded', 'failed')
                 GROUP BY revision, holdout, outcome",
            )
            .context("prepare the outcome counts")?;
        for row in counts
            .query_map([procedure], |row| {
                Ok((
                    row.get::<_, u32>(0)?,
                    row.get::<_, bool>(1)?,
                    row.get::<_, bool>(2)?,
                    row.get::<_, u32>(3)?,
                ))
            })
            .context("count outcomes")?
        {
            let (revision, holdout, succeeded, count) = row.context("read an outcome count")?;
            let add = |evidence: &mut Evidence| {
                *evidence = if succeeded {
                    Evidence::new(evidence.successes + count, evidence.failures)
                } else {
                    Evidence::new(evidence.successes, evidence.failures + count)
                };
            };
            if holdout {
                add(&mut evidence.holdout);
            } else if let Some((_, _, revision_evidence)) = evidence
                .revisions
                .iter_mut()
                .find(|(r, _, _)| *r == revision)
            {
                add(revision_evidence);
            }
        }
        Ok(evidence)
    }

    pub fn refresh_outcomes(&mut self, procedure: &str) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start refreshing outcomes")?;
        let rows: Vec<(i64, u32, String)> = {
            let mut statement = tx
                .prepare("SELECT rowid, revision, document FROM procedures WHERE id = ?1")
                .context("prepare the revision lookup")?;
            statement
                .query_map([procedure], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })
                .context("look up revisions")?
                .collect::<Result<_, _>>()
                .context("read revisions")?
        };
        let mut counts = tx
            .prepare(
                "SELECT
                    coalesce(sum(NOT holdout AND outcome = 'succeeded'), 0),
                    coalesce(sum(NOT holdout AND outcome = 'failed'), 0),
                    coalesce(sum(NOT holdout), 0),
                    coalesce(sum(holdout), 0),
                    max(CASE WHEN NOT holdout AND outcome = 'succeeded' THEN at END)
                 FROM injections WHERE procedure = ?1 AND revision = ?2",
            )
            .context("prepare the outcome totals")?;
        for (rowid, revision, document) in rows {
            let (successes, failures, injections, holdouts, last_success): (
                u32,
                u32,
                u32,
                u32,
                Option<String>,
            ) = counts
                .query_row(params![procedure, revision], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })
                .context("total a revision's outcomes")?;
            let mut stored: Procedure =
                serde_json::from_str(&document).context("parse a stored procedure")?;
            stored.outcomes.successes = successes;
            stored.outcomes.failures = failures;
            stored.outcomes.injections = injections;
            stored.outcomes.holdouts = holdouts;
            stored.outcomes.last_success_at = last_success
                .map(|at| at.parse())
                .transpose()
                .context("parse a success time")?;
            tx.execute(
                "UPDATE procedures SET document = ?1 WHERE rowid = ?2",
                params![Self::document(&stored)?, rowid],
            )
            .context("store a revision's outcomes")?;
        }
        drop(counts);
        tx.commit().context("commit outcomes")
    }

    pub fn outcome_summaries(&self) -> Result<Vec<OutcomeSummary>> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT i.procedure,
                    coalesce((SELECT title FROM procedures p WHERE p.id = i.procedure
                              ORDER BY p.state IN ('active', 'stale') DESC, p.revision DESC LIMIT 1), ''),
                    coalesce(sum(NOT i.holdout AND i.outcome = 'succeeded'), 0),
                    coalesce(sum(NOT i.holdout AND i.outcome = 'failed'), 0),
                    coalesce(sum(i.holdout AND i.outcome = 'succeeded'), 0),
                    coalesce(sum(i.holdout AND i.outcome = 'failed'), 0),
                    avg(CASE WHEN NOT i.holdout AND i.outcome IN ('succeeded', 'failed') THEN e.tool_calls END),
                    avg(CASE WHEN i.holdout AND i.outcome IN ('succeeded', 'failed') THEN e.tool_calls END)
                 FROM injections i
                 LEFT JOIN extractions e ON e.session = i.session AND e.first_seq = i.task
                 GROUP BY i.procedure ORDER BY count(*) DESC, i.procedure",
            )
            .context("prepare the outcome summary")?;
        statement
            .query_map([], |row| {
                Ok(OutcomeSummary {
                    procedure: row.get(0)?,
                    title: row.get(1)?,
                    injected: Evidence::new(row.get(2)?, row.get(3)?),
                    held_out: Evidence::new(row.get(4)?, row.get(5)?),
                    injected_tool_calls: row.get(6)?,
                    held_out_tool_calls: row.get(7)?,
                })
            })
            .context("summarize outcomes")?
            .collect::<Result<_, _>>()
            .context("read outcome summaries")
    }

    pub fn servable_revisions(&self, procedure: &str) -> Result<Vec<crate::ProcedureRow>> {
        let mut statement = self
            .conn
            .prepare_cached(
                "SELECT rowid, document FROM procedures
                 WHERE id = ?1 AND state IN ('active', 'stale', 'candidate') ORDER BY revision",
            )
            .context("prepare the servable revision lookup")?;
        let rows = statement
            .query_map([procedure], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .context("look up servable revisions")?;
        rows.map(|row| {
            let (rowid, document) = row.context("read a servable revision")?;
            let procedure = serde_json::from_str(&document).context("parse a stored procedure")?;
            Ok(crate::ProcedureRow { rowid, procedure })
        })
        .collect()
    }

    pub fn set_states(&mut self, procedure: &str, changes: &[(u32, Lifecycle)]) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start changing states")?;
        for (revision, state) in changes {
            let document: Option<(i64, String)> = tx
                .query_row(
                    "SELECT rowid, document FROM procedures WHERE id = ?1 AND revision = ?2",
                    params![procedure, revision],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .optional()
                .context("read a revision")?;
            let Some((rowid, document)) = document else {
                continue;
            };
            let mut stored: Procedure =
                serde_json::from_str(&document).context("parse a stored procedure")?;
            stored.state = *state;
            tx.execute(
                "UPDATE procedures SET state = ?1, document = ?2 WHERE rowid = ?3",
                params![Self::state_name(*state), Self::document(&stored)?, rowid],
            )
            .context("change a revision's state")?;
        }
        Self::reindex(&tx, procedure)?;
        tx.commit().context("commit state changes")
    }

    pub fn retire(&mut self, procedure: &str) -> Result<usize> {
        let revisions: Vec<u32> = self
            .revisions(procedure)?
            .into_iter()
            .map(|row| row.procedure.revision)
            .collect();
        let changes: Vec<(u32, Lifecycle)> = revisions
            .iter()
            .map(|revision| (*revision, Lifecycle::Retired))
            .collect();
        self.set_states(procedure, &changes)?;
        Ok(revisions.len())
    }

    pub fn usage(&self) -> Result<Vec<Usage>> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT id, revision, state, coalesce(used_at, updated_at) FROM procedures
                 WHERE state IN ('active', 'stale', 'candidate') ORDER BY id, revision",
            )
            .context("prepare the usage listing")?;
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, u32>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .context("list usage")?;
        rows.map(|row| {
            let (procedure, revision, state, used_at) = row.context("read usage")?;
            Ok(Usage {
                procedure,
                revision,
                state: Self::parse_state(&state),
                used_at: used_at.parse().context("parse a usage time")?,
            })
        })
        .collect()
    }
}
