use trodden_core::{
    Procedure, SessionId,
    procedure::{Lifecycle, Source, Step, StepKind},
};
use trodden_learn::Evidence;
use trodden_store::{Cue, ExtractionRecord, Forget, Injection, Progress, Store, Upsert};

const REPO: &str = "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c";

#[derive(Debug)]
struct Fixture;

impl Fixture {
    fn pagination() -> Procedure {
        let mut procedure: Procedure = Procedure::example();
        procedure.state = Lifecycle::Candidate;
        procedure
    }

    fn from_session(mut procedure: Procedure, session: &str) -> Procedure {
        procedure.provenance.sources[0].session = SessionId::new(session);
        procedure
    }

    fn checked_with(mut procedure: Procedure, command: &str) -> Procedure {
        let check = procedure
            .steps
            .iter_mut()
            .rfind(|step| step.kind == StepKind::Verify)
            .expect("the example has a check");
        check.command = Some(command.to_owned());
        procedure
    }

    fn injection(session: &str, revision: u32, holdout: bool) -> Injection {
        Injection {
            session: session.to_owned(),
            procedure: "p_7f3a91c2".to_owned(),
            revision,
            holdout,
            cue: Cue::Prompt,
            at: "2026-09-22T09:00:00Z".parse().expect("valid timestamp"),
        }
    }
}

#[test]
fn identical_steps_refresh_and_different_steps_revise() {
    let mut store = Store::open_in_memory().expect("store opens");

    let first = store.upsert(&Fixture::pagination()).expect("stored");
    let again = store
        .upsert(&Fixture::from_session(
            Fixture::pagination(),
            "9d3e7a10-5b2c-4e8f-a1d6-0c7b9e2f4a58",
        ))
        .expect("stored");
    let mut different = Fixture::from_session(
        Fixture::pagination(),
        "2c8f1b6e-7d4a-4c3e-9b0f-5e1a3d7c9b24",
    );
    different.steps.insert(
        0,
        Step {
            kind: StepKind::Setup,
            command: Some("npm ci".to_owned()),
            target: None,
            symbols: Vec::new(),
            reads: Vec::new(),
            writes: Vec::new(),
        },
    );
    let revised = store.upsert(&different).expect("stored");

    assert!(matches!(first, Upsert::Created { .. }));
    assert_eq!(
        again,
        Upsert::Refreshed {
            rowid: first.rowid()
        }
    );
    assert!(matches!(revised, Upsert::Revised { .. }));

    let revisions = store.revisions("p_7f3a91c2").expect("revisions read");
    let states: Vec<_> = revisions
        .iter()
        .map(|row| (row.procedure.revision, row.procedure.state))
        .collect();
    assert_eq!(states, [(1, Lifecycle::Active), (2, Lifecycle::Candidate)]);
    let sources: Vec<&Source> = revisions[0].procedure.provenance.sources.iter().collect();
    assert_eq!(
        sources.len(),
        2,
        "the refresh added its session as a source"
    );
}

#[test]
fn literals_that_differ_between_sessions_become_slots() {
    let mut store = Store::open_in_memory().expect("store opens");

    let first = store
        .upsert(&Fixture::checked_with(
            Fixture::pagination(),
            "npm test -- page_overlap",
        ))
        .expect("stored");
    let second = store
        .upsert(&Fixture::checked_with(
            Fixture::from_session(
                Fixture::pagination(),
                "9d3e7a10-5b2c-4e8f-a1d6-0c7b9e2f4a58",
            ),
            "npm test -- total_pages",
        ))
        .expect("stored");
    let third = store
        .upsert(&Fixture::checked_with(
            Fixture::from_session(
                Fixture::pagination(),
                "2c8f1b6e-7d4a-4c3e-9b0f-5e1a3d7c9b24",
            ),
            "npm test -- cursor_repeat",
        ))
        .expect("stored");
    let flagged = store
        .upsert(&Fixture::checked_with(
            Fixture::from_session(
                Fixture::pagination(),
                "6a0e4f2d-8c1b-4d7e-b3a9-1f5c7e9d2b40",
            ),
            "npm test -- --watch",
        ))
        .expect("stored");

    assert_eq!(
        second,
        Upsert::Generalized {
            rowid: first.rowid()
        }
    );
    assert_eq!(
        third,
        Upsert::Generalized {
            rowid: first.rowid()
        }
    );
    assert!(
        matches!(flagged, Upsert::Revised { .. }),
        "a different flag is a different command"
    );
    let revisions = store.revisions("p_7f3a91c2").expect("revisions read");
    let merged = &revisions[0].procedure;
    let check = merged
        .steps
        .iter()
        .rfind(|step| step.kind == StepKind::Verify)
        .and_then(|step| step.command.as_deref());
    assert_eq!(check, Some("npm test -- {test}"));
    let slot = merged
        .slots
        .iter()
        .find(|slot| slot.name == "test")
        .expect("the check gained a slot");
    assert_eq!(
        slot.examples,
        ["page_overlap", "total_pages", "cursor_repeat"]
    );
    assert_eq!(merged.provenance.sources.len(), 3);
}

#[test]
fn finds_procedures_by_entity_and_text() {
    let mut store = Store::open_in_memory().expect("store opens");
    let rowid = store
        .upsert(&Fixture::pagination())
        .expect("stored")
        .rowid();

    let by_stem = store
        .entity_hits(&["paginate".to_owned()], REPO)
        .expect("entities read");
    let other_repo = store
        .entity_hits(&["paginate".to_owned()], "0000")
        .expect("entities read");
    let by_text = store
        .lexical_hits(
            &[
                "page".to_owned(),
                "duplicate".to_owned(),
                "repeats".to_owned(),
            ],
            REPO,
            5,
        )
        .expect("text searched");

    assert!(
        by_stem
            .iter()
            .any(|hit| hit.rowid == rowid && hit.kind == "path"),
        "{by_stem:?}"
    );
    assert!(
        other_repo.is_empty(),
        "procedures are scoped to their repository"
    );
    assert_eq!(by_text.first().map(|hit| hit.rowid), Some(rowid));
}

#[test]
fn lexical_search_treats_prompt_syntax_as_text() {
    let mut store = Store::open_in_memory().expect("store opens");
    store.upsert(&Fixture::pagination()).expect("stored");

    let hits = store
        .lexical_hits(
            &[
                "\"quoted\"".to_owned(),
                "-flag".to_owned(),
                "a:b".to_owned(),
                "npm*".to_owned(),
            ],
            REPO,
            5,
        )
        .expect("odd terms do not break the query");

    assert!(hits.len() <= 1);
}

#[test]
fn forgetting_removes_procedures_and_their_index_entries() {
    let mut store = Store::open_in_memory().expect("store opens");
    store.upsert(&Fixture::pagination()).expect("stored");

    let deleted = store
        .forget(Forget::Procedure("p_7f3a91c2"))
        .expect("forgotten");

    assert_eq!(deleted, 1);
    assert!(
        store
            .entity_hits(&["paginate".to_owned()], REPO)
            .expect("entities read")
            .is_empty()
    );
    assert!(
        store
            .lexical_hits(&["pagination".to_owned()], REPO, 5)
            .expect("text searched")
            .is_empty()
    );
}

#[test]
fn ingest_records_are_idempotent() {
    let store = Store::open_in_memory().expect("store opens");
    let session = "0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90";
    let record = ExtractionRecord {
        session: session.to_owned(),
        first_seq: 0,
        summary: "Explain the pagination module".to_owned(),
        procedure: None,
        rejection: Some("no files were changed".to_owned()),
        outcome: Some("unjudged".to_owned()),
        tool_calls: Some(4),
        span: Some((
            "2026-09-21T14:02:11Z".to_owned(),
            "2026-09-21T14:02:40Z".to_owned(),
        )),
        at: "2026-09-21T14:02:44Z".to_owned(),
    };

    store.record_extraction(&record).expect("recorded");
    store
        .record_extraction(&record)
        .expect("recording again is harmless");
    let progress = Progress {
        extracted_through: Some(5),
        ended: true,
    };
    store
        .set_progress(
            session,
            "claude-code",
            "/home/dev/t.jsonl",
            Some(REPO),
            progress,
            &record.at,
        )
        .expect("saved");

    assert_eq!(store.rejections(10).expect("rejections read"), [record]);
    assert_eq!(
        store.progress(session).expect("progress read"),
        Some(progress)
    );
    assert_eq!(store.stats().expect("stats read").rejections, 1);
}

#[test]
fn saves_progress_before_any_task_is_extracted() {
    let store = Store::open_in_memory().expect("store opens");
    let session = "0667cb29-ae94-4131-92e9-2d6b8112d0df";
    let progress = Progress {
        extracted_through: None,
        ended: false,
    };

    store
        .set_progress(
            session,
            "claude-code",
            "/home/dev/t.jsonl",
            Some(REPO),
            progress,
            "2026-09-30T02:20:00Z",
        )
        .expect("progress saves");

    assert_eq!(
        store.progress(session).expect("progress read"),
        Some(progress)
    );
}

#[test]
fn records_injections_per_session() {
    let store = Store::open_in_memory().expect("store opens");
    let session = "0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90";

    store
        .record_injection(&Fixture::injection(session, 1, false))
        .expect("recorded");

    assert!(store.was_injected(session, "p_7f3a91c2").expect("read"));
    assert!(
        !store
            .was_injected("another-session", "p_7f3a91c2")
            .expect("read")
    );
}

#[test]
fn settled_outcomes_count_per_revision_and_holdout() {
    let mut store = Store::open_in_memory().expect("store opens");
    store.upsert(&Fixture::pagination()).expect("stored");
    let outcomes = [
        ("s1", false, "succeeded"),
        ("s2", false, "succeeded"),
        ("s3", false, "failed"),
        ("s4", false, "unjudged"),
        ("s5", true, "failed"),
    ];
    for (session, holdout, _) in outcomes {
        store
            .record_injection(&Fixture::injection(session, 1, holdout))
            .expect("recorded");
    }
    for (record, (_, _, outcome)) in store
        .injections(None)
        .expect("injections read")
        .iter()
        .zip(outcomes)
    {
        store.settle_injection(record, 0, outcome).expect("settled");
    }

    let evidence = store.family_evidence("p_7f3a91c2").expect("evidence read");
    store
        .refresh_outcomes("p_7f3a91c2")
        .expect("outcomes refresh");

    assert_eq!(evidence.injected(), Evidence::new(2, 1));
    assert_eq!(evidence.holdout, Evidence::new(0, 1));
    let revision = &store.revisions("p_7f3a91c2").expect("revisions read")[0].procedure;
    assert_eq!(
        (
            revision.outcomes.successes,
            revision.outcomes.failures,
            revision.outcomes.injections,
            revision.outcomes.holdouts
        ),
        (2, 1, 4, 1)
    );
}

#[test]
fn promotion_moves_the_index_to_the_new_incumbent() {
    let mut store = Store::open_in_memory().expect("store opens");
    store.upsert(&Fixture::pagination()).expect("stored");
    let mut different = Fixture::from_session(
        Fixture::pagination(),
        "2c8f1b6e-7d4a-4c3e-9b0f-5e1a3d7c9b24",
    );
    different.steps.remove(0);
    let revised = store.upsert(&different).expect("stored").rowid();

    store
        .set_states(
            "p_7f3a91c2",
            &[(2, Lifecycle::Active), (1, Lifecycle::Candidate)],
        )
        .expect("states change");

    let hits = store
        .entity_hits(&["npm test".to_owned()], REPO)
        .expect("entities read");
    assert!(hits.iter().all(|hit| hit.rowid == revised), "{hits:?}");
    assert_eq!(
        store
            .servable_revisions("p_7f3a91c2")
            .expect("revisions read")
            .len(),
        2
    );
}

#[test]
fn refreshing_learns_prompts_slot_values_and_lessons() {
    let mut store = Store::open_in_memory().expect("store opens");
    store.upsert(&Fixture::pagination()).expect("stored");
    let mut again = Fixture::from_session(
        Fixture::pagination(),
        "9d3e7a10-5b2c-4e8f-a1d6-0c7b9e2f4a58",
    );
    again.trigger.text = "The second page shows the first page's last item again".to_owned();
    again.avoid = vec!["`npm run test:all` failed (exit 1); `npm test` worked".to_owned()];

    store.upsert(&again).expect("stored");

    let stored = &store.revisions("p_7f3a91c2").expect("revisions read")[0].procedure;
    assert_eq!(stored.trigger.examples, [again.trigger.text.clone()]);
    assert!(stored.avoid.contains(&again.avoid[0]));
    let found = store
        .lexical_hits(&["item".to_owned(), "second".to_owned()], REPO, 5)
        .expect("text searched");
    assert_eq!(found.len(), 1, "the new prompt is searchable");
}

#[test]
fn retired_families_stay_out_of_recall() {
    let mut store = Store::open_in_memory().expect("store opens");
    store.upsert(&Fixture::pagination()).expect("stored");

    assert_eq!(store.retire("p_7f3a91c2").expect("retired"), 1);
    let mut later = Fixture::from_session(
        Fixture::pagination(),
        "2c8f1b6e-7d4a-4c3e-9b0f-5e1a3d7c9b24",
    );
    later.steps.remove(0);
    store.upsert(&later).expect("stored");

    assert!(
        store
            .entity_hits(&["paginate".to_owned()], REPO)
            .expect("entities read")
            .is_empty()
    );
    let states: Vec<Lifecycle> = store
        .revisions("p_7f3a91c2")
        .expect("revisions read")
        .iter()
        .map(|row| row.procedure.state)
        .collect();
    assert_eq!(states, [Lifecycle::Retired, Lifecycle::Retired]);
}

#[test]
fn stores_several_embeddings_per_revision() {
    let mut store = Store::open_in_memory().expect("store opens");
    let rowid = store
        .upsert(&Fixture::pagination())
        .expect("stored")
        .rowid();

    store
        .set_embeddings(rowid, &[vec![1, 2], vec![3, 4]])
        .expect("embeddings store");
    store
        .set_embeddings(rowid, &[vec![5, 6]])
        .expect("embeddings replace");

    assert_eq!(
        store.recallable_embeddings().expect("embeddings read"),
        [(rowid, REPO.to_owned(), vec![5, 6])]
    );
}
