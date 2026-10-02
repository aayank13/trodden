use super::{
    Condition, Entity, Lifecycle, Outcomes, Procedure, Provenance, Resource, Scope, Source, Step,
    StepKind, Trigger, Verification,
};
use crate::{FamilyId, HarnessId, ProcedureId, RepoId, SessionId};

impl Procedure {
    pub fn example() -> Self {
        let at = |text: &str| text.parse().expect("literal is a valid timestamp");
        Self {
            id: ProcedureId::new("p_7f3a91c2"),
            family: FamilyId::new("f_pagination_offset"),
            revision: 3,
            title: "Fix an off-by-one in catalog pagination".to_owned(),
            scope: Scope::Repo {
                repo: RepoId::new("4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c"),
            },
            trigger: Trigger {
                entities: vec![
                    Entity::Path("src/paginate.js".to_owned()),
                    Entity::Symbol("paginate".to_owned()),
                    Entity::Command("npm test".to_owned()),
                ],
                text: "pagination page repeats duplicate item off by one listing".to_owned(),
                examples: Vec::new(),
            },
            preconditions: vec![
                Condition::SymbolInFile {
                    path: "src/paginate.js".to_owned(),
                    symbol: "paginate".to_owned(),
                },
                Condition::ProgramOnPath {
                    program: "node".to_owned(),
                },
            ],
            steps: vec![
                Step {
                    kind: StepKind::Edit,
                    command: None,
                    target: Some("src/paginate.js".to_owned()),
                    symbols: vec!["paginate".to_owned()],
                    reads: Vec::new(),
                    writes: vec![Resource::File("src/paginate.js".to_owned())],
                },
                Step {
                    kind: StepKind::Verify,
                    command: Some("npm test".to_owned()),
                    target: None,
                    symbols: Vec::new(),
                    reads: vec![Resource::File("fixtures/seed.csv".to_owned())],
                    writes: vec![Resource::Artifact("fixtures/data.json".to_owned())],
                },
            ],
            verify: Some(Verification {
                command: "npm test".to_owned(),
                expect_exit: 0,
                declared_by: None,
            }),
            avoid: vec![
                "`node --test` directly fails: the fixtures are built by `npm test`".to_owned(),
            ],
            slots: Vec::new(),
            provenance: Provenance {
                sources: vec![Source {
                    session: SessionId::new("0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90"),
                    harness: HarnessId::new("claude-code"),
                    model: Some("claude-sonnet-5-5".to_owned()),
                    commit: Some("9f2c4e1a7b3d5f60a8c1e2d3b4a5968778695a4b".to_owned()),
                }],
                created_at: at("2026-09-21T14:03:02Z"),
                updated_at: at("2026-09-27T09:41:55Z"),
            },
            outcomes: Outcomes {
                successes: 9,
                failures: 1,
                injections: 10,
                holdouts: 1,
                last_success_at: Some(at("2026-09-27T09:41:20Z")),
            },
            state: Lifecycle::Active,
        }
    }
}
