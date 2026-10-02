mod artifact;
mod envelope;
mod index;
mod kind;
mod skeleton;

use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::Path,
};

use anyhow::{Context, Result};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use trodden_core::{
    Procedure,
    procedure::{Condition, Lifecycle},
};
use trodden_embed::Embedder;
use trodden_learn::{Evidence, Holdout, Policy};
use trodden_store::{ProcedureRow, Store, Terms};

pub use artifact::{Artifact, Artifacts};
pub use envelope::Envelope;
pub use index::{Neighbor, VectorIndex};
pub use kind::TaskKind;
pub use skeleton::Skeleton;

const STAGE_LIMIT: usize = 10;

const MAX_PROMPT_BYTES: usize = 16 * 1024;

const MAX_TERMS: usize = 512;

const RRF_K: f64 = 60.0;

#[derive(Debug, Clone)]
pub struct Query<'a> {
    pub prompt: &'a str,
    pub repo: &'a str,
    pub root: &'a Path,
    pub session: Option<&'a str>,
}

impl Query<'_> {
    fn place(&self) -> Place<'_> {
        Place {
            root: self.root,
            session: self.session,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ErrorQuery<'a> {
    pub signature: &'a str,
    pub repo: &'a str,
    pub root: &'a Path,
    pub session: Option<&'a str>,
}

#[derive(Debug, Clone, Copy)]
struct Place<'a> {
    root: &'a Path,
    session: Option<&'a str>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Signals {
    pub exact: Vec<(String, String)>,
    pub lexical: Option<(usize, f64)>,
    pub semantic: Option<(usize, f32)>,
    pub cosine: Option<f32>,
    pub fused: f64,
    pub same_kind: Option<bool>,
    pub same_object: Option<bool>,
}

impl Signals {
    fn exact_weight(&self) -> f64 {
        self.exact
            .iter()
            .map(|(_, kind)| match kind.as_str() {
                "path" => 2.0,
                "symbol" => 1.5,
                _ => 1.0,
            })
            .sum()
    }

    fn ranked_cosine(&self) -> f32 {
        self.semantic.map_or(0.0, |(_, cosine)| cosine)
    }

    fn is_confident(&self, state: Lifecycle) -> bool {
        let cosine = self.ranked_cosine();
        let top_lexical = self.lexical.is_some_and(|(rank, _)| rank == 0);
        if state == Lifecycle::Stale {
            return self.exact_weight() >= 2.0 && top_lexical && cosine >= Gate::STALE_COSINE;
        }
        if self.same_kind == Some(false) || self.same_object == Some(false) {
            return cosine >= Gate::STRONG_COSINE;
        }
        let on_topic = self
            .cosine
            .is_none_or(|cosine| cosine >= Gate::EXACT_FLOOR_COSINE);
        (self.exact_weight() >= 2.0
            && on_topic
            && (top_lexical || cosine >= Gate::SUPPORTED_COSINE))
            || cosine >= Gate::STRONG_COSINE
            || (top_lexical && cosine >= Gate::SUPPORTED_COSINE)
            || (top_lexical && self.same_kind == Some(true) && cosine >= Gate::SAME_KIND_COSINE)
    }
}

#[derive(Debug)]
struct Gate;

impl Gate {
    const STRONG_COSINE: f32 = 0.8;
    const SUPPORTED_COSINE: f32 = 0.45;
    const SAME_KIND_COSINE: f32 = 0.30;
    const EXACT_FLOOR_COSINE: f32 = 0.25;
    const STALE_COSINE: f32 = 0.6;
    const AMBIGUITY_RATIO: f64 = 0.95;
}

#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub row: ProcedureRow,
    pub signals: Signals,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Abstention {
    NoCandidates,
    NotConfident,
    Ambiguous,
    PreconditionFailed(String),
    AlreadyInjected,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    Inject(Box<Match>),
    Withhold(Box<Match>),
    Abstain(Abstention),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub decision: Decision,
    pub candidates: Vec<Match>,
}

#[derive(Debug)]
pub struct Recall<'a> {
    store: &'a Store,
    semantic: Option<(Embedder, VectorIndex)>,
    rng: SmallRng,
}

impl<'a> Recall<'a> {
    pub fn new(store: &'a Store, semantic: Option<(Embedder, VectorIndex)>) -> Self {
        Self {
            store,
            semantic,
            rng: rand::make_rng(),
        }
    }

    pub fn seeded(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn recall(&mut self, query: &Query<'_>) -> Result<Outcome> {
        let candidates = self.candidates(query)?;
        let decision = self.decide(query.place(), &candidates)?;
        Ok(Outcome {
            decision,
            candidates,
        })
    }

    pub fn recall_error(&mut self, query: &ErrorQuery<'_>) -> Result<Outcome> {
        let mut candidates = Vec::new();
        for hit in self
            .store
            .entity_hits(&[query.signature.to_lowercase()], query.repo)?
            .into_iter()
            .filter(|hit| hit.kind == "error")
        {
            if let Some(row) = self.store.procedure(hit.rowid)? {
                candidates.push(Match {
                    row,
                    signals: Signals {
                        exact: vec![(hit.key, hit.kind)],
                        ..Signals::default()
                    },
                });
            }
        }
        let place = Place {
            root: query.root,
            session: query.session,
        };
        let decision = match candidates.as_slice() {
            [] => Decision::Abstain(Abstention::NoCandidates),
            [best] => self.serve(place, best)?,
            _ => Decision::Abstain(Abstention::Ambiguous),
        };
        Ok(Outcome {
            decision,
            candidates,
        })
    }

    fn candidates(&mut self, query: &Query<'_>) -> Result<Vec<Match>> {
        let prompt = Self::bounded(query.prompt);
        let terms = Self::terms(prompt);
        let mut signals: HashMap<i64, Signals> = HashMap::new();

        let mut keys = terms.clone();
        for term in terms.iter().filter(|term| term.contains('/')) {
            let name = term.rsplit('/').next().unwrap_or(term);
            keys.push(name.to_owned());
            if let Some((stem, _)) = name.split_once('.').filter(|(stem, _)| stem.len() > 2) {
                keys.push(stem.to_owned());
            }
        }
        keys.sort_unstable();
        keys.dedup();
        for hit in self.store.entity_hits(&keys, query.repo)? {
            let entry = signals.entry(hit.rowid).or_default();
            if !entry.exact.iter().any(|(key, _)| *key == hit.key) {
                entry.exact.push((hit.key, hit.kind));
            }
        }
        for (rank, hit) in self
            .store
            .lexical_hits(&terms, query.repo, STAGE_LIMIT)?
            .into_iter()
            .enumerate()
        {
            signals.entry(hit.rowid).or_default().lexical = Some((rank, hit.score));
        }
        if let Some((embedder, index)) = &mut self.semantic
            && let Some(embedding) = embedder
                .embed(Skeleton::of(prompt).as_str())
                .context("embed the prompt")?
        {
            for (rank, neighbor) in index
                .search(&embedding, query.repo, STAGE_LIMIT)?
                .into_iter()
                .enumerate()
            {
                let entry = signals.entry(neighbor.rowid).or_default();
                entry.semantic = Some((rank, neighbor.cosine));
                entry.cosine = Some(neighbor.cosine);
            }
            let unranked: Vec<i64> = signals
                .iter()
                .filter(|(_, s)| s.semantic.is_none())
                .map(|(rowid, _)| *rowid)
                .collect();
            for neighbor in index.cosines(&embedding, query.repo, &unranked)? {
                signals
                    .get_mut(&neighbor.rowid)
                    .expect("looked up rows have signals")
                    .cosine = Some(neighbor.cosine);
            }
        }

        let mut exact: Vec<(i64, f64)> = signals
            .iter()
            .filter(|(_, s)| !s.exact.is_empty())
            .map(|(rowid, s)| (*rowid, s.exact_weight()))
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (rank, (rowid, _)) in exact.into_iter().enumerate() {
            signals
                .get_mut(&rowid)
                .expect("ranked rows have signals")
                .fused += Self::rrf(rank);
        }
        for s in signals.values_mut() {
            s.fused += s.lexical.map_or(0.0, |(rank, _)| Self::rrf(rank));
            s.fused += s.semantic.map_or(0.0, |(rank, _)| Self::rrf(rank));
        }

        let query_kind = TaskKind::of(prompt);
        let requested = Artifacts::requested(prompt);
        let mut candidates = Vec::with_capacity(signals.len());
        for (rowid, mut signals) in signals {
            if let Some(row) = self.store.procedure(rowid)? {
                let trigger = &row.procedure.trigger;
                signals.same_kind = query_kind
                    .zip(TaskKind::of(&trigger.text))
                    .map(|(query, procedure)| query == procedure);
                let learned = Artifacts::mentioned(
                    std::iter::once(trigger.text.as_str())
                        .chain(trigger.examples.iter().map(String::as_str)),
                );
                signals.same_object = (!requested.is_empty() && !learned.is_empty())
                    .then(|| requested.intersects(learned));
                candidates.push(Match { row, signals });
            }
        }
        candidates.sort_by(|a, b| {
            b.signals
                .fused
                .total_cmp(&a.signals.fused)
                .then(a.row.rowid.cmp(&b.row.rowid))
        });
        Ok(candidates)
    }

    fn bounded(prompt: &str) -> &str {
        &prompt[..prompt.floor_char_boundary(MAX_PROMPT_BYTES)]
    }

    fn terms(prompt: &str) -> Vec<String> {
        let terms = Terms::of(prompt);
        let mut seen = HashSet::new();
        terms
            .iter()
            .filter(|term| seen.insert(term.as_str()))
            .take(MAX_TERMS)
            .cloned()
            .collect()
    }

    fn rrf(rank: usize) -> f64 {
        let rank = rank as f64;
        1.0 / (RRF_K + rank + 1.0)
    }

    fn decide(&mut self, place: Place<'_>, candidates: &[Match]) -> Result<Decision> {
        let Some(best) = candidates.first() else {
            return Ok(Decision::Abstain(Abstention::NoCandidates));
        };
        if !best.signals.is_confident(best.row.procedure.state) {
            return Ok(Decision::Abstain(Abstention::NotConfident));
        }
        let ambiguous = candidates.iter().skip(1).any(|rival| {
            rival.row.procedure.family != best.row.procedure.family
                && rival.signals.fused >= best.signals.fused * Gate::AMBIGUITY_RATIO
                && rival.signals.exact_weight() >= best.signals.exact_weight()
        });
        if ambiguous {
            return Ok(Decision::Abstain(Abstention::Ambiguous));
        }
        self.serve(place, best)
    }

    fn serve(&mut self, place: Place<'_>, best: &Match) -> Result<Decision> {
        let id = best.row.procedure.id.as_str();
        if let Some(session) = place.session
            && self.store.was_injected(session, id)?
        {
            return Ok(Decision::Abstain(Abstention::AlreadyInjected));
        }

        let mut revisions = self.store.servable_revisions(id)?;
        if revisions.is_empty() {
            revisions.push(best.row.clone());
        }
        let mut first_failure = None;
        revisions.retain(
            |row| match Preconditions::first_failure(&row.procedure, place.root) {
                Some(failure) => {
                    if row.rowid == best.row.rowid || first_failure.is_none() {
                        first_failure = Some(failure);
                    }
                    false
                }
                None => true,
            },
        );
        let evidence: Vec<Evidence> = revisions
            .iter()
            .map(|row| Evidence::of(&row.procedure.outcomes))
            .collect();
        let Some(chosen) = Policy::choose(&evidence, &mut self.rng) else {
            let reason = first_failure.unwrap_or_else(|| "no revision applies".to_owned());
            return Ok(Decision::Abstain(Abstention::PreconditionFailed(reason)));
        };
        let chosen = Match {
            row: revisions.swap_remove(chosen),
            signals: best.signals.clone(),
        };

        if place.session.is_some() {
            let family = self.store.family_evidence(id)?;
            let rate = Holdout::rate(
                self.store.holdout_rate()?,
                family.injected(),
                family.holdout,
            );
            if self.rng.random::<f64>() < rate {
                return Ok(Decision::Withhold(Box::new(chosen)));
            }
        }
        Ok(Decision::Inject(Box::new(chosen)))
    }
}

#[derive(Debug)]
struct Preconditions;

impl Preconditions {
    fn first_failure(procedure: &Procedure, root: &Path) -> Option<String> {
        procedure
            .preconditions
            .iter()
            .find_map(|condition| match condition {
                Condition::FileExists { path } => {
                    (!root.join(path).exists()).then(|| format!("{path} does not exist"))
                }
                Condition::SymbolInFile { path, symbol } => fs::read_to_string(root.join(path))
                    .map_or(true, |text| !text.contains(symbol.as_str()))
                    .then(|| format!("{path} no longer mentions {symbol}")),
                Condition::ProgramOnPath { program } => {
                    (!Self::on_path(program)).then(|| format!("{program} is not on PATH"))
                }
                Condition::EnvVarSet { name } => env::var_os(name)
                    .is_none()
                    .then(|| format!("{name} is not set")),
                _ => None,
            })
    }

    fn on_path(program: &str) -> bool {
        if program.contains('/') {
            return Path::new(program).exists();
        }
        env::var_os("PATH").is_some_and(|path| {
            env::split_paths(&path).any(|dir| {
                let candidate = dir.join(program);
                candidate.is_file() || (cfg!(windows) && candidate.with_extension("exe").is_file())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use trodden_core::{FamilyId, Procedure, ProcedureId, procedure::Trigger};
    use trodden_embed::{DIMS, Embedding, Quantized};

    use super::*;

    const REPO: &str = "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c";

    const PROMPT: &str =
        "Page 2 of the listing repeats the last item. Fix the pagination in src/paginate.js.";

    const CRASH_PROMPT: &str = "Fix the crash in src/paginate.js when the user logs out.";

    fn outcome(store: &Store, prompt: &str) -> Outcome {
        Recall::new(store, None)
            .seeded(7)
            .recall(&Query {
                prompt,
                repo: REPO,
                root: Path::new("."),
                session: None,
            })
            .expect("recall succeeds")
    }

    fn direction(weights: &[(usize, f32)]) -> Embedding {
        let mut embedding = [0.0; DIMS];
        for (axis, weight) in weights {
            embedding[*axis] = *weight;
        }
        embedding
    }

    fn semantic(name: &str, vectors: &[(i64, Embedding)]) -> (Embedder, VectorIndex) {
        let dir =
            std::env::temp_dir().join(format!("trodden-recall-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).expect("scratch directory is writable");
        let token = b"crash";
        let mut pack = b"TRDEMB\x01\0".to_vec();
        for field in [DIMS, 1, token.len(), 0, 0, token.len(), 0] {
            pack.extend(
                u32::try_from(field)
                    .expect("pack field is small")
                    .to_le_bytes(),
            );
        }
        pack.extend(token);
        pack.extend(Quantized::new(&direction(&[(0, 1.0)])).to_bytes());
        let pack_path = dir.join("embeddings.pack");
        fs::write(&pack_path, pack).expect("pack is writable");
        let rows: Vec<(i64, String, Vec<u8>)> = vectors
            .iter()
            .map(|(rowid, vector)| (*rowid, REPO.to_owned(), Quantized::new(vector).to_bytes()))
            .collect();
        let index_path = dir.join("recall.index");
        VectorIndex::build(&rows, &index_path).expect("index builds");
        (
            Embedder::open(&pack_path).expect("pack opens"),
            VectorIndex::open(&index_path).expect("index opens"),
        )
    }

    fn unrelated(n: usize) -> Procedure {
        let mut procedure = Procedure::example();
        procedure.id = ProcedureId::new(format!("p_rotate{n}"));
        procedure.family = FamilyId::new(format!("f_rotate{n}"));
        procedure.title = format!("Rotate the staging credentials of service {n}");
        procedure.trigger = Trigger {
            entities: Vec::new(),
            text: "rotate staging credentials".to_owned(),
            examples: Vec::new(),
        };
        procedure.preconditions.clear();
        procedure.steps.clear();
        procedure.verify = None;
        procedure.avoid.clear();
        procedure
    }

    fn semantic_outcome(
        name: &str,
        prompt: &str,
        target: Option<Embedding>,
        closer: usize,
    ) -> Outcome {
        let mut store = Store::open_in_memory().expect("in-memory store opens");
        let mut procedure = Procedure::example();
        procedure.preconditions.clear();
        let rowid = store.upsert(&procedure).expect("procedure stores").rowid();
        let mut vectors: Vec<(i64, Embedding)> =
            target.map(|target| (rowid, target)).into_iter().collect();
        for n in 0..closer {
            let rowid = store
                .upsert(&unrelated(n))
                .expect("procedure stores")
                .rowid();
            vectors.push((rowid, direction(&[(0, 0.5), (n + 2, 0.866)])));
        }
        Recall::new(&store, Some(semantic(name, &vectors)))
            .seeded(7)
            .recall(&Query {
                prompt,
                repo: REPO,
                root: Path::new("."),
                session: None,
            })
            .expect("recall succeeds")
    }

    fn semantic_decision(
        name: &str,
        prompt: &str,
        target: Option<Embedding>,
        closer: usize,
    ) -> Decision {
        semantic_outcome(name, prompt, target, closer).decision
    }

    #[test]
    fn abstains_on_a_far_exact_match_however_many_procedures_are_closer() {
        let far = Some(direction(&[(1, 1.0)]));

        assert_eq!(
            semantic_decision("far-alone", CRASH_PROMPT, far, 0),
            Decision::Abstain(Abstention::NotConfident)
        );
        assert_eq!(
            semantic_decision("far-crowded", CRASH_PROMPT, far, 12),
            Decision::Abstain(Abstention::NotConfident)
        );
    }

    #[test]
    fn injects_a_near_exact_match_among_other_procedures() {
        let near = Some(direction(&[(0, 0.6), (1, 0.8)]));

        assert!(matches!(
            semantic_decision("near-crowded", CRASH_PROMPT, near, 12),
            Decision::Inject(_)
        ));
    }

    #[test]
    fn injects_a_moderately_close_exact_match_outside_the_semantic_top_ten() {
        let moderate = Some(direction(&[(0, 0.35), (1, 0.937)]));

        let outcome = semantic_outcome("moderate-crowded", CRASH_PROMPT, moderate, 12);

        let Decision::Inject(chosen) = outcome.decision else {
            panic!("expected an injection, got {:?}", outcome.decision);
        };
        assert_eq!(chosen.signals.semantic, None);
        let cosine = chosen.signals.cosine.expect("real cosine is looked up");
        assert!((cosine - 0.35).abs() < 0.01);
    }

    #[test]
    fn falls_back_to_exact_matches_without_a_semantic_score() {
        let far = Some(direction(&[(1, 1.0)]));
        let unembeddable = "Fix the hang in src/paginate.js when the user logs out.";

        assert!(matches!(
            semantic_decision("unembedded", unembeddable, far, 12),
            Decision::Inject(_)
        ));
        assert!(matches!(
            semantic_decision("unindexed", CRASH_PROMPT, None, 12),
            Decision::Inject(_)
        ));
    }

    fn scored(rowid: i64, family: &str, fused: f64, exact: &[&str]) -> Match {
        let mut procedure = Procedure::example();
        procedure.id = ProcedureId::new(format!("p_{family}"));
        procedure.family = FamilyId::new(family);
        procedure.preconditions.clear();
        Match {
            row: ProcedureRow { rowid, procedure },
            signals: Signals {
                exact: exact
                    .iter()
                    .map(|key| ((*key).to_owned(), "path".to_owned()))
                    .collect(),
                fused,
                ..Signals::default()
            },
        }
    }

    fn gate(candidates: &[Match]) -> Decision {
        let store = Store::open_in_memory().expect("in-memory store opens");
        let place = Place {
            root: Path::new("."),
            session: None,
        };
        Recall::new(&store, None)
            .seeded(7)
            .decide(place, candidates)
            .expect("decision succeeds")
    }

    #[test]
    fn abstains_when_any_other_family_is_as_strong() {
        let mut best = scored(1, "f_best", 1.0, &["src/paginate.js"]);
        best.signals.lexical = Some((0, 1.0));
        let weak = scored(2, "f_weak", 0.99, &[]);
        let strong = scored(3, "f_strong", 0.97, &["src/paginate.js"]);
        let distant = scored(4, "f_distant", 0.9, &["src/paginate.js"]);

        let ambiguous = Decision::Abstain(Abstention::Ambiguous);
        assert_eq!(
            gate(&[best.clone(), weak.clone(), strong.clone()]),
            ambiguous
        );
        assert_eq!(gate(&[best.clone(), strong]), ambiguous);
        assert!(matches!(
            gate(&[best.clone(), weak.clone()]),
            Decision::Inject(_)
        ));
        assert!(matches!(gate(&[best, weak, distant]), Decision::Inject(_)));
    }

    #[test]
    fn recalls_the_same_despite_a_huge_paste() {
        let mut store = Store::open_in_memory().expect("in-memory store opens");
        let mut procedure = Procedure::example();
        procedure.preconditions.clear();
        store.upsert(&procedure).expect("procedure stores");
        let paste: String = (0..200_000).map(|n| format!("w{n:x} ")).collect();

        let expected = outcome(&store, PROMPT);

        assert!(matches!(expected.decision, Decision::Inject(_)));
        assert_eq!(outcome(&store, &format!("{PROMPT}\n{paste}")), expected);
    }

    #[test]
    fn cuts_long_prompts_on_a_char_boundary() {
        let long = "€".repeat(MAX_PROMPT_BYTES);

        assert_eq!(Recall::bounded(PROMPT), PROMPT);
        assert_eq!(Recall::bounded(&long).len(), MAX_PROMPT_BYTES / 3 * 3);
    }

    #[test]
    fn keeps_the_first_unique_terms() {
        let words: String = (0..2 * MAX_TERMS).map(|n| format!("w{n} w{n} ")).collect();

        assert_eq!(
            Recall::terms("fix src/a.rs then fix src/a.rs"),
            ["fix", "src/a.rs"]
        );
        let terms = Recall::terms(&words);
        assert_eq!(terms.len(), MAX_TERMS);
        assert_eq!(terms[MAX_TERMS - 1], format!("w{}", MAX_TERMS - 1));
    }
}
