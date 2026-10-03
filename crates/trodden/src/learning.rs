use std::collections::{BTreeMap, BTreeSet};

use anyhow::Result;
use jiff::{SignedDuration, Timestamp};
use trodden_core::{
    Trace,
    procedure::Lifecycle,
    trace::{Event, EventKind},
};
use trodden_extract::TaskOutcome;
use trodden_learn::{Aging, Policy, RevisionEvidence};
use trodden_store::{Cue, FamilyEvidence, InjectionRecord, Store};

const PROMPT_WINDOW: SignedDuration = SignedDuration::from_mins(2);

const FAILURE_SLACK: SignedDuration = SignedDuration::from_secs(5);

#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskSpan {
    pub(crate) first_seq: u32,
    pub(crate) last_seq: u32,
    pub(crate) outcome: TaskOutcome,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Changes {
    pub(crate) promoted: usize,
    pub(crate) quarantined: usize,
    pub(crate) aged: usize,
}

impl Changes {
    pub(crate) fn any(&self) -> bool {
        *self != Self::default()
    }

    fn count(&mut self, decided: &[(u32, Lifecycle)]) {
        for (_, state) in decided {
            match state {
                Lifecycle::Quarantined => self.quarantined += 1,
                Lifecycle::Active => self.promoted += 1,
                _ => {}
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct Learning<'a> {
    store: &'a mut Store,
    settled: BTreeMap<String, Vec<InjectionRecord>>,
}

impl<'a> Learning<'a> {
    pub(crate) fn new(store: &'a mut Store) -> Self {
        Self {
            store,
            settled: BTreeMap::new(),
        }
    }

    pub(crate) fn settle_injections(
        &mut self,
        trace: &Trace,
        tasks: &[TaskSpan],
    ) -> Result<(usize, BTreeSet<String>)> {
        for record in self.store.injections(Some(trace.session.as_str()))? {
            if record.outcome.is_some() {
                continue;
            }
            let Some(seq) = Self::landing(&trace.events, &record) else {
                continue;
            };
            if let Some(task) = tasks
                .iter()
                .find(|task| (task.first_seq..=task.last_seq).contains(&seq))
            {
                self.settled
                    .entry(record.injection.procedure.clone())
                    .or_default()
                    .push(InjectionRecord {
                        task: Some(task.first_seq),
                        outcome: Some(task.outcome.as_str().to_owned()),
                        ..record
                    });
            }
        }
        let settled = self.settled.values().map(Vec::len).sum();
        Ok((settled, self.settled.keys().cloned().collect()))
    }

    fn landing(events: &[Event], record: &InjectionRecord) -> Option<u32> {
        let at = record.injection.at;
        match record.injection.cue {
            Cue::Prompt => events
                .iter()
                .filter(|event| matches!(event.kind, EventKind::Prompt { .. }))
                .map(|event| (event.at.duration_since(at).abs(), event.seq))
                .filter(|(gap, _)| *gap <= PROMPT_WINDOW)
                .min()
                .map(|(_, seq)| seq),
            Cue::Failure => events
                .iter()
                .rev()
                .find(|event| event.at <= at + FAILURE_SLACK)
                .map(|event| event.seq),
        }
    }

    pub(crate) fn settle(&mut self, procedures: &BTreeSet<String>) -> Result<Changes> {
        let mut changes = Changes::default();
        for procedure in procedures {
            let settled = self.settled.remove(procedure).unwrap_or_default();
            let decided = self
                .store
                .settle_procedure(procedure, &settled, &[], Self::decide)?;
            changes.count(&decided);
        }
        Ok(changes)
    }

    fn decide(family: &FamilyEvidence) -> Vec<(u32, Lifecycle)> {
        let revisions: Vec<RevisionEvidence> = family
            .revisions
            .iter()
            .map(|(revision, state, evidence)| RevisionEvidence {
                revision: *revision,
                state: *state,
                evidence: *evidence,
            })
            .collect();
        Policy::settle(&revisions, family.holdout)
    }

    pub(crate) fn age(&mut self, now: Timestamp) -> Result<Changes> {
        let mut by_procedure: BTreeMap<String, Vec<(u32, Lifecycle)>> = BTreeMap::new();
        for usage in self.store.usage()? {
            let state = Aging::state(usage.state, usage.used_at, now);
            if state != usage.state {
                by_procedure
                    .entry(usage.procedure)
                    .or_default()
                    .push((usage.revision, state));
            }
        }
        let mut changes = Changes::default();
        for (procedure, aged) in &by_procedure {
            let decided = self
                .store
                .settle_procedure(procedure, &[], aged, Self::decide)?;
            changes.aged += aged.len();
            changes.count(&decided);
        }
        Ok(changes)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use rand::{RngExt, SeedableRng, rngs::SmallRng};
    use trodden_core::{HarnessId, Procedure, SessionId, procedure::Step};
    use trodden_recall::{Decision, Query, Recall};
    use trodden_store::Injection;

    use super::*;

    const PROMPT: &str = "Page 2 in src/paginate.js repeats the last product from page 1";
    const REPO: &str = "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c";

    fn procedure(session: &str) -> Procedure {
        let mut procedure: Procedure = Procedure::example();
        procedure.preconditions.clear();
        procedure.outcomes = trodden_core::procedure::Outcomes::default();
        procedure.provenance.sources[0].session = SessionId::new(session);
        procedure
    }

    fn simulate(
        store: &mut Store,
        success: &BTreeMap<u32, f64>,
        baseline: f64,
        rounds: usize,
        seed: u64,
    ) -> Vec<Option<u32>> {
        let mut world = SmallRng::seed_from_u64(seed);
        let mut served = Vec::new();
        for round in 0..rounds {
            let session = format!("session-{round}");
            let decision = Recall::new(store, None)
                .seeded(seed + round as u64)
                .recall(&Query {
                    prompt: PROMPT,
                    repo: REPO,
                    root: Path::new("/"),
                    session: Some(&session),
                })
                .expect("recall runs")
                .decision;
            let (chosen, holdout) = match decision {
                Decision::Inject(chosen) => (chosen, false),
                Decision::Withhold(chosen) => (chosen, true),
                Decision::Abstain(_) => {
                    served.push(None);
                    continue;
                }
            };
            let revision = chosen.row.procedure.revision;
            served.push((!holdout).then_some(revision));
            store
                .record_injection(&Injection {
                    session: session.clone(),
                    procedure: chosen.row.procedure.id.to_string(),
                    revision,
                    holdout,
                    cue: Cue::Prompt,
                    at: Timestamp::UNIX_EPOCH,
                })
                .expect("recorded");
            let rate = if holdout {
                baseline
            } else {
                success[&revision]
            };
            let outcome = if world.random::<f64>() < rate {
                TaskOutcome::Succeeded
            } else {
                TaskOutcome::Failed
            };
            let record = store
                .injections(Some(&session))
                .expect("injections read")
                .remove(0);
            store
                .settle_injection(&record, 0, outcome.as_str())
                .expect("settled");
            let touched = BTreeSet::from([record.injection.procedure]);
            Learning::new(store).settle(&touched).expect("settles");
        }
        served
    }

    #[test]
    fn selection_converges_to_the_better_revision() {
        let mut store = Store::open_in_memory().expect("store opens");
        store.upsert(&procedure("teacher-1")).expect("stored");
        let mut better = procedure("teacher-2");
        better.steps.insert(
            0,
            Step {
                kind: trodden_core::procedure::StepKind::Setup,
                command: Some("npm ci".to_owned()),
                target: None,
                symbols: Vec::new(),
                reads: Vec::new(),
                writes: Vec::new(),
            },
        );
        store.upsert(&better).expect("stored");
        store.set_holdout_rate(Some(0.0)).expect("holdout off");
        let success = BTreeMap::from([(1, 0.3), (2, 0.8)]);

        let served = simulate(&mut store, &success, 0.5, 60, 11);

        let states: Vec<(u32, Lifecycle)> = store
            .revisions("p_7f3a91c2")
            .expect("revisions read")
            .iter()
            .map(|row| (row.procedure.revision, row.procedure.state))
            .collect();
        assert_eq!(states, [(1, Lifecycle::Candidate), (2, Lifecycle::Active)]);
        let late = &served[40..];
        let good = late.iter().filter(|revision| **revision == Some(2)).count();
        assert!(good >= 16, "{late:?}");
    }

    #[test]
    fn a_recipe_that_does_worse_than_none_is_quarantined() {
        let mut store = Store::open_in_memory().expect("store opens");
        store.upsert(&procedure("teacher-1")).expect("stored");
        store.set_holdout_rate(Some(0.3)).expect("holdout set");
        let success = BTreeMap::from([(1, 0.1)]);

        let served = simulate(&mut store, &success, 0.8, 80, 5);

        let state = store.revisions("p_7f3a91c2").expect("revisions read")[0]
            .procedure
            .state;
        assert_eq!(state, Lifecycle::Quarantined);
        assert!(
            served.ends_with(&[None]),
            "quarantined procedures are not served"
        );
    }

    #[test]
    fn an_interrupted_ingest_still_settles_the_policy_on_the_next_run() {
        let mut store = Store::open_in_memory().expect("store opens");
        store.upsert(&procedure("teacher-1")).expect("stored");
        let at: Timestamp = "2026-10-01T10:00:00Z".parse().expect("timestamp is valid");
        let session = |round: usize| format!("session-{round}");
        for round in 0..5 {
            store
                .record_injection(&Injection {
                    session: session(round),
                    procedure: "p_7f3a91c2".to_owned(),
                    revision: 1,
                    holdout: false,
                    cue: Cue::Prompt,
                    at,
                })
                .expect("recorded");
        }
        let trace = |round: usize| Trace {
            session: SessionId::new(session(round)),
            harness: HarnessId::new("claude-code"),
            model: None,
            cwd: "/".to_owned(),
            commit: None,
            started_at: at,
            events: vec![Event {
                seq: 0,
                at,
                kind: EventKind::Prompt {
                    summary: PROMPT.to_owned(),
                },
            }],
        };
        let tasks = [TaskSpan {
            first_seq: 0,
            last_seq: 0,
            outcome: TaskOutcome::Failed,
        }];
        for round in 0..4 {
            let mut learning = Learning::new(&mut store);
            let (_, touched) = learning
                .settle_injections(&trace(round), &tasks)
                .expect("injections match");
            learning.settle(&touched).expect("settles");
        }

        Learning::new(&mut store)
            .settle_injections(&trace(4), &tasks)
            .expect("injections match");
        let mut learning = Learning::new(&mut store);
        let (settled, touched) = learning
            .settle_injections(&trace(4), &tasks)
            .expect("injections match");
        let changes = learning.settle(&touched).expect("settles");

        assert_eq!((settled, changes.quarantined), (1, 1));
        let stored = &store.revisions("p_7f3a91c2").expect("revisions read")[0].procedure;
        assert_eq!(
            (stored.state, stored.outcomes.failures),
            (Lifecycle::Quarantined, 5)
        );
    }

    #[test]
    fn unused_procedures_go_stale_then_leave_recall() {
        let mut store = Store::open_in_memory().expect("store opens");
        store.upsert(&procedure("teacher-1")).expect("stored");
        let days = |days: i64| Timestamp::now() + SignedDuration::from_hours(days * 24);
        let recallable = |store: &Store| {
            !store
                .entity_hits(&["paginate".to_owned()], REPO)
                .expect("entities read")
                .is_empty()
        };
        let state = |store: &Store| {
            store.revisions("p_7f3a91c2").expect("read")[0]
                .procedure
                .state
        };

        let month = Learning::new(&mut store).age(days(40)).expect("ages");
        assert_eq!((month.aged, state(&store)), (1, Lifecycle::Stale));
        assert!(recallable(&store), "stale procedures stay recallable");

        Learning::new(&mut store).age(days(100)).expect("ages");
        assert_eq!(state(&store), Lifecycle::Archived);
        assert!(!recallable(&store));
    }
}
