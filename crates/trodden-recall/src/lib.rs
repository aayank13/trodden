mod artifact;
mod envelope;
mod index;

use std::{
    cell::OnceCell,
    collections::{HashMap, HashSet},
    env,
    fs::{self, File},
    io::Read,
    path::{Component, Path},
};

use anyhow::{Context, Result};
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use trodden_capture::ErrorSignature;
use trodden_core::{
    Procedure, Skeleton, TaskKind,
    procedure::{Condition, Lifecycle},
};
use trodden_embed::Embedder;
use trodden_learn::{Evidence, Holdout, Policy};
use trodden_redact::Redactor;
use trodden_store::{ProcedureRow, Store, Terms};

pub use artifact::{Artifact, Artifacts};
pub use envelope::Envelope;
pub use index::{Neighbor, VectorIndex};

const STAGE_LIMIT: usize = 10;

const SEMANTIC_OVERFETCH: usize = 5;

const MAX_PROMPT_BYTES: usize = 16 * 1024;

const MAX_TERMS: usize = 512;

const MAX_PHRASE_WORDS: usize = 3;

const MAX_ERROR_SIGNATURES: usize = 8;

const RRF_K: f64 = 60.0;

const MAX_FILE_BYTES: u64 = 1024 * 1024;

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
        self.exact.iter().map(|(_, kind)| self.weight(kind)).sum()
    }

    fn weight(&self, kind: &str) -> f64 {
        match kind {
            "path" | "error" => 2.0,
            "stem" if self.cosine_confirms_topic() => 2.0,
            "stem" => 0.5,
            "symbol" => 1.5,
            _ => 1.0,
        }
    }

    fn cosine_confirms_topic(&self) -> bool {
        self.cosine
            .is_some_and(|cosine| cosine >= Gate::EXACT_FLOOR_COSINE)
    }

    fn add_exact(&mut self, key: String, kind: String) {
        let weight = self.weight(&kind);
        match self.exact.iter().position(|(known, _)| *known == key) {
            Some(index) if weight > self.weight(&self.exact[index].1) => self.exact[index].1 = kind,
            Some(_) => {}
            None => self.exact.push((key, kind)),
        }
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
    pub semantic_error: Option<String>,
}

#[derive(Debug)]
pub struct Recall<'a> {
    store: &'a Store,
    semantic: Option<(Embedder, VectorIndex)>,
    semantic_error: Option<String>,
    rng: SmallRng,
}

impl<'a> Recall<'a> {
    pub fn new(store: &'a Store, semantic: Option<(Embedder, VectorIndex)>) -> Self {
        Self {
            store,
            semantic,
            semantic_error: None,
            rng: rand::make_rng(),
        }
    }

    pub fn semantic_unavailable(mut self, error: &anyhow::Error) -> Self {
        self.semantic = None;
        self.semantic_error = Some(format!("{error:#}"));
        self
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
            semantic_error: self.semantic_error.clone(),
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
            if let Some(row) = self.store.recallable_procedure(hit.rowid, query.repo)? {
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
            semantic_error: None,
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
        keys.extend(Self::phrases(prompt));
        keys.extend(PastedErrors::signatures(prompt));
        keys.sort_unstable();
        keys.dedup();
        let hits = self.store.entity_hits(&keys, query.repo)?;
        for hit in &hits {
            signals.entry(hit.rowid).or_default();
        }
        for (rank, hit) in self
            .store
            .lexical_hits(&terms, query.repo, STAGE_LIMIT)?
            .into_iter()
            .enumerate()
        {
            signals.entry(hit.rowid).or_default().lexical = Some((rank, hit.score));
        }
        let known: Vec<i64> = signals.keys().copied().collect();
        match self.semantic_neighbors(prompt, query.repo, &known) {
            Ok(Some((nearest, looked_up))) => {
                for neighbor in looked_up {
                    signals
                        .get_mut(&neighbor.rowid)
                        .expect("looked up rows have signals")
                        .cosine = Some(neighbor.cosine);
                }
                for (rank, neighbor) in self
                    .recallable_neighbors(nearest, query.repo)?
                    .into_iter()
                    .enumerate()
                {
                    let entry = signals.entry(neighbor.rowid).or_default();
                    entry.semantic = Some((rank, neighbor.cosine));
                    entry.cosine = Some(neighbor.cosine);
                }
            }
            Ok(None) => {}
            Err(error) => {
                self.semantic = None;
                self.semantic_error = Some(format!("{error:#}"));
            }
        }

        let mut matches: HashMap<i64, Match> = HashMap::with_capacity(signals.len());
        for (rowid, signals) in signals {
            if let Some(row) = self.store.recallable_procedure(rowid, query.repo)? {
                matches.insert(rowid, Match { row, signals });
            }
        }
        for hit in hits {
            if let Some(candidate) = matches.get_mut(&hit.rowid) {
                let kind = if hit.is_stem_of(&candidate.row.procedure) {
                    "stem".to_owned()
                } else {
                    hit.kind
                };
                candidate.signals.add_exact(hit.key, kind);
            }
        }

        let mut exact: Vec<(i64, f64)> = matches
            .iter()
            .filter(|(_, candidate)| !candidate.signals.exact.is_empty())
            .map(|(rowid, candidate)| (*rowid, candidate.signals.exact_weight()))
            .collect();
        exact.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (rank, (rowid, _)) in exact.into_iter().enumerate() {
            matches
                .get_mut(&rowid)
                .expect("ranked rows are candidates")
                .signals
                .fused += Self::rrf(rank);
        }

        let query_kind = TaskKind::of(prompt);
        let requested = Artifacts::requested(prompt);
        for Match { row, signals } in matches.values_mut() {
            signals.fused += signals.lexical.map_or(0.0, |(rank, _)| Self::rrf(rank));
            signals.fused += signals.semantic.map_or(0.0, |(rank, _)| Self::rrf(rank));
            let trigger = &row.procedure.trigger;
            signals.same_kind = query_kind
                .zip(trigger.kind())
                .map(|(query, procedure)| query == procedure);
            let learned = Artifacts::mentioned(
                std::iter::once(trigger.text.as_str())
                    .chain(trigger.examples.iter().map(String::as_str)),
            );
            signals.same_object = (!requested.is_empty() && !learned.is_empty())
                .then(|| requested.intersects(learned));
        }
        let mut candidates: Vec<Match> = matches.into_values().collect();
        candidates.sort_by(|a, b| {
            b.signals
                .fused
                .total_cmp(&a.signals.fused)
                .then(a.row.rowid.cmp(&b.row.rowid))
        });
        Ok(candidates)
    }

    fn semantic_neighbors(
        &mut self,
        prompt: &str,
        repo: &str,
        known: &[i64],
    ) -> Result<Option<(Vec<Neighbor>, Vec<Neighbor>)>> {
        let Some((embedder, index)) = &mut self.semantic else {
            return Ok(None);
        };
        let Some(embedding) = embedder
            .embed(Skeleton::of(prompt).as_str())
            .context("embed the prompt")?
        else {
            return Ok(None);
        };
        let nearest = index
            .search(&embedding, repo, STAGE_LIMIT * SEMANTIC_OVERFETCH)
            .context("search the vector index")?;
        let looked_up = index
            .cosines(&embedding, repo, known)
            .context("look up cosines in the vector index")?;
        Ok(Some((nearest, looked_up)))
    }

    fn recallable_neighbors(&self, nearest: Vec<Neighbor>, repo: &str) -> Result<Vec<Neighbor>> {
        let mut neighbors = Vec::with_capacity(STAGE_LIMIT);
        for neighbor in nearest {
            if self.store.is_recallable(neighbor.rowid, repo)? {
                neighbors.push(neighbor);
                if neighbors.len() == STAGE_LIMIT {
                    break;
                }
            }
        }
        Ok(neighbors)
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

    fn phrases(prompt: &str) -> Vec<String> {
        let words = Terms::of(prompt);
        let words = words.as_slice();
        let mut seen = HashSet::new();
        (0..words.len())
            .flat_map(|start| {
                (2..=MAX_PHRASE_WORDS).filter_map(move |length| words.get(start..start + length))
            })
            .map(|phrase| phrase.join(" "))
            .filter(|phrase| seen.insert(phrase.clone()))
            .take(MAX_TERMS)
            .collect()
    }

    fn rrf(rank: usize) -> f64 {
        let rank = rank as f64;
        1.0 / (RRF_K + rank + 1.0)
    }

    fn decide(&mut self, place: Place<'_>, candidates: &[Match]) -> Result<Decision> {
        let Some(top) = candidates.first() else {
            return Ok(Decision::Abstain(Abstention::NoCandidates));
        };
        let best = candidates
            .iter()
            .take_while(|candidate| {
                candidate.signals.fused >= top.signals.fused * Gate::AMBIGUITY_RATIO
            })
            .find(|candidate| candidate.signals.same_kind == Some(true))
            .unwrap_or(top);
        if !best.signals.is_confident(best.row.procedure.state) {
            return Ok(Decision::Abstain(Abstention::NotConfident));
        }
        let ambiguous = candidates.iter().any(|rival| {
            rival.row.procedure.family != best.row.procedure.family
                && !(best.signals.same_kind == Some(true) && rival.signals.same_kind == Some(false))
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
            return Ok(Decision::Abstain(Abstention::NoCandidates));
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
struct PastedErrors;

impl PastedErrors {
    const MARKERS: &[&str] = &[
        "error",
        "exception",
        "panic",
        "fatal",
        "err!",
        "command not found",
        "no such file or directory",
        "cannot find module",
        "not recognized as",
        "permission denied",
        "module not found",
        "unresolved import",
        "undefined reference",
    ];

    fn signatures(prompt: &str) -> Vec<String> {
        let redactor = OnceCell::new();
        let mut seen = HashSet::new();
        prompt
            .lines()
            .filter(|line| Self::may_be_error(line))
            .filter_map(|line| ErrorSignature::of(line, redactor.get_or_init(Redactor::new)))
            .filter(|signature| seen.insert(signature.clone()))
            .take(MAX_ERROR_SIGNATURES)
            .collect()
    }

    fn may_be_error(line: &str) -> bool {
        let lower = line.to_ascii_lowercase();
        line.contains('\x1b')
            || Self::MARKERS.iter().any(|marker| lower.contains(marker))
            || (0..line.len()).any(|start| Self::located_at(&line.as_bytes()[start..]))
    }

    fn located_at(rest: &[u8]) -> bool {
        let (separator, close) = match rest.first() {
            Some(b':') => (b':', None),
            Some(b'(') => (b',', Some(b')')),
            _ => return false,
        };
        let line = Self::digits(&rest[1..]);
        let column = rest.get(line + 2..).map_or(0, Self::digits);
        line > 0
            && column > 0
            && rest.get(line + 1) == Some(&separator)
            && close.is_none_or(|close| rest.get(line + column + 2) == Some(&close))
    }

    fn digits(bytes: &[u8]) -> usize {
        bytes
            .iter()
            .take_while(|byte| byte.is_ascii_digit())
            .count()
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
                Condition::SymbolInFile { path, symbol } => Self::read(root, path)
                    .is_none_or(|text| !text.contains(symbol.as_str()))
                    .then(|| format!("{path} no longer mentions {symbol}")),
                Condition::ProgramOnPath { program } => {
                    (!Self::on_path(program, root)).then(|| format!("{program} is not on PATH"))
                }
                Condition::EnvVarSet { name } => env::var_os(name)
                    .is_none()
                    .then(|| format!("{name} is not set")),
                _ => None,
            })
    }

    fn read(root: &Path, path: &str) -> Option<String> {
        let relative = Path::new(path);
        if !relative
            .components()
            .all(|part| matches!(part, Component::Normal(_) | Component::CurDir))
        {
            return None;
        }
        let path = root.join(relative).canonicalize().ok()?;
        if !path.starts_with(root.canonicalize().ok()?) || !fs::metadata(&path).ok()?.is_file() {
            return None;
        }
        let mut bytes = Vec::new();
        File::open(&path)
            .ok()?
            .take(MAX_FILE_BYTES)
            .read_to_end(&mut bytes)
            .ok()?;
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn on_path(program: &str, root: &Path) -> bool {
        let program = Path::new(program);
        if program.components().nth(1).is_some() {
            return Self::runnable(&root.join(program));
        }
        env::var_os("PATH").is_some_and(|path| {
            env::split_paths(&path).any(|dir| Self::runnable(&dir.join(program)))
        })
    }

    fn runnable(candidate: &Path) -> bool {
        candidate.is_file()
            || (cfg!(windows)
                && env::var_os("PATHEXT")
                    .unwrap_or_else(|| ".COM;.EXE;.BAT;.CMD".into())
                    .to_string_lossy()
                    .split(';')
                    .filter(|extension| !extension.is_empty())
                    .any(|extension| {
                        let mut name = candidate.as_os_str().to_owned();
                        name.push(extension);
                        Path::new(&name).is_file()
                    }))
    }
}

#[cfg(test)]
mod tests {
    use trodden_core::{
        FamilyId, Procedure, ProcedureId, RepoId,
        procedure::{Entity, Scope, Trigger},
    };
    use trodden_embed::{DIMS, Embedding, Quantized};

    use super::*;

    const REPO: &str = "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c";

    const PROMPT: &str =
        "Page 2 of the listing repeats the last item. Fix the pagination in src/paginate.js.";

    const CRASH_PROMPT: &str = "Fix the crash in src/paginate.js when the user logs out.";

    #[derive(Debug)]
    struct Fixture;

    impl Fixture {
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
            pack.extend(Quantized::new(&Self::direction(&[(0, 1.0)])).to_bytes());
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
                    .upsert(&Self::unrelated(n))
                    .expect("procedure stores")
                    .rowid();
                vectors.push((rowid, Self::direction(&[(0, 0.5), (n + 2, 0.866)])));
            }
            Recall::new(&store, Some(Self::semantic(name, &vectors)))
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
            Self::semantic_outcome(name, prompt, target, closer).decision
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
            let mut store = Store::open_in_memory().expect("in-memory store opens");
            for candidate in candidates {
                store
                    .upsert(&candidate.row.procedure)
                    .expect("procedure stores");
            }
            let place = Place {
                root: Path::new("."),
                session: None,
            };
            Recall::new(&store, None)
                .seeded(7)
                .decide(place, candidates)
                .expect("decision succeeds")
        }
    }

    #[test]
    fn abstains_on_a_far_exact_match_however_many_procedures_are_closer() {
        let far = Some(Fixture::direction(&[(1, 1.0)]));

        assert_eq!(
            Fixture::semantic_decision("far-alone", CRASH_PROMPT, far, 0),
            Decision::Abstain(Abstention::NotConfident)
        );
        assert_eq!(
            Fixture::semantic_decision("far-crowded", CRASH_PROMPT, far, 12),
            Decision::Abstain(Abstention::NotConfident)
        );
    }

    #[test]
    fn injects_a_near_exact_match_among_other_procedures() {
        let near = Some(Fixture::direction(&[(0, 0.6), (1, 0.8)]));

        assert!(matches!(
            Fixture::semantic_decision("near-crowded", CRASH_PROMPT, near, 12),
            Decision::Inject(_)
        ));
    }

    #[test]
    fn injects_a_moderately_close_exact_match_outside_the_semantic_top_ten() {
        let moderate = Some(Fixture::direction(&[(0, 0.35), (1, 0.937)]));

        let outcome = Fixture::semantic_outcome("moderate-crowded", CRASH_PROMPT, moderate, 12);

        let Decision::Inject(chosen) = outcome.decision else {
            panic!("expected an injection, got {:?}", outcome.decision);
        };
        assert_eq!(chosen.signals.semantic, None);
        let cosine = chosen.signals.cosine.expect("real cosine is looked up");
        assert!((cosine - 0.35).abs() < 0.01);
    }

    #[test]
    fn falls_back_to_exact_matches_without_a_semantic_score() {
        let far = Some(Fixture::direction(&[(1, 1.0)]));
        let unembeddable = "Fix the hang in src/paginate.js when the user logs out.";

        assert!(matches!(
            Fixture::semantic_decision("unembedded", unembeddable, far, 12),
            Decision::Inject(_)
        ));
        assert!(matches!(
            Fixture::semantic_decision("unindexed", CRASH_PROMPT, None, 12),
            Decision::Inject(_)
        ));
    }

    #[derive(Debug)]
    struct StaleIndex {
        store: Store,
        rowid: i64,
    }

    impl StaleIndex {
        fn indexed() -> Self {
            let mut store = Store::open_in_memory().expect("in-memory store opens");
            let mut procedure = Procedure::example();
            procedure.preconditions.clear();
            let rowid = store.upsert(&procedure).expect("procedure stores").rowid();
            Self { store, rowid }
        }

        fn recall(&self, name: &str) -> Outcome {
            let vectors = [(self.rowid, Fixture::direction(&[(0, 1.0)]))];
            Recall::new(&self.store, Some(Fixture::semantic(name, &vectors)))
                .seeded(7)
                .recall(&Query {
                    prompt: CRASH_PROMPT,
                    repo: REPO,
                    root: Path::new("."),
                    session: None,
                })
                .expect("recall succeeds")
        }
    }

    #[derive(Debug)]
    struct Truncated;

    impl Truncated {
        fn recall(name: &str, file: &str) -> Outcome {
            let stale = StaleIndex::indexed();
            let vectors = [(stale.rowid, Fixture::direction(&[(0, 1.0)]))];
            let semantic = Fixture::semantic(name, &vectors);
            let path = std::env::temp_dir()
                .join(format!("trodden-recall-{name}-{}", std::process::id()))
                .join(file);
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .expect("scratch file opens")
                .set_len(16)
                .expect("scratch file truncates");
            Recall::new(&stale.store, Some(semantic))
                .seeded(7)
                .recall(&Query {
                    prompt: CRASH_PROMPT,
                    repo: REPO,
                    root: Path::new("."),
                    session: None,
                })
                .expect("recall succeeds without the semantic stage")
        }
    }

    #[test]
    fn recalls_exact_and_lexical_matches_when_the_semantic_stage_fails() {
        for (name, file, stage) in [
            ("truncated-index", "recall.index", "search the vector index"),
            ("truncated-pack", "embeddings.pack", "embed the prompt"),
        ] {
            let outcome = Truncated::recall(name, file);

            let Decision::Inject(chosen) = &outcome.decision else {
                panic!("{name}: expected an injection, got {:?}", outcome.decision);
            };
            assert_eq!(chosen.signals.semantic, None, "{name}");
            assert_eq!(chosen.signals.cosine, None, "{name}");
            assert!(!chosen.signals.exact.is_empty(), "{name}");
            let error = outcome.semantic_error.expect("the failure is reported");
            assert!(error.starts_with(stage), "{name}: {error}");
        }
    }

    #[test]
    fn reports_no_semantic_error_when_the_semantic_stage_works() {
        assert_eq!(StaleIndex::indexed().recall("healthy").semantic_error, None);
    }

    #[test]
    fn carries_a_semantic_open_error_into_the_outcome() {
        let stale = StaleIndex::indexed();

        let outcome = Recall::new(&stale.store, None)
            .semantic_unavailable(&anyhow::anyhow!("recall.index is truncated"))
            .seeded(7)
            .recall(&Query {
                prompt: CRASH_PROMPT,
                repo: REPO,
                root: Path::new("."),
                session: None,
            })
            .expect("recall succeeds");

        assert!(matches!(outcome.decision, Decision::Inject(_)));
        assert_eq!(
            outcome.semantic_error.as_deref(),
            Some("recall.index is truncated")
        );
    }

    #[test]
    fn abstains_on_a_procedure_that_left_recall_after_the_index_was_built() {
        assert!(matches!(
            StaleIndex::indexed().recall("stale-active").decision,
            Decision::Inject(_)
        ));
        for state in [Lifecycle::Retired, Lifecycle::Quarantined] {
            let mut stale = StaleIndex::indexed();
            stale
                .store
                .set_states("p_7f3a91c2", &[(1, state)])
                .expect("state changes");

            let outcome = stale.recall(&format!("stale-{state:?}"));

            assert_eq!(
                outcome.decision,
                Decision::Abstain(Abstention::NoCandidates),
                "{state:?}"
            );
            assert!(outcome.candidates.is_empty(), "{state:?}");
        }
    }

    #[test]
    fn ignores_a_reused_rowid_that_now_belongs_to_another_repository() {
        let mut stale = StaleIndex::indexed();
        stale
            .store
            .forget(trodden_store::Forget::Procedure("p_7f3a91c2"))
            .expect("procedure is forgotten");
        let mut elsewhere = Fixture::unrelated(0);
        elsewhere.scope = Scope::Repo {
            repo: RepoId::new("9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d"),
        };
        let reused = stale.store.upsert(&elsewhere).expect("procedure stores");
        assert_eq!(reused.rowid(), stale.rowid);

        let outcome = stale.recall("stale-elsewhere");

        assert_eq!(
            outcome.decision,
            Decision::Abstain(Abstention::NoCandidates)
        );
        assert!(outcome.candidates.is_empty());
    }

    #[test]
    fn never_serves_a_match_without_a_servable_revision() {
        let mut stale = StaleIndex::indexed();
        let row = stale
            .store
            .procedure(stale.rowid)
            .expect("procedure reads")
            .expect("procedure exists");
        stale.store.retire("p_7f3a91c2").expect("procedure retires");
        let retired = Match {
            row,
            signals: Signals {
                semantic: Some((0, 1.0)),
                cosine: Some(1.0),
                ..Signals::default()
            },
        };
        let place = Place {
            root: Path::new("."),
            session: None,
        };

        let decision = Recall::new(&stale.store, None)
            .seeded(7)
            .decide(place, &[retired])
            .expect("decision succeeds");

        assert_eq!(decision, Decision::Abstain(Abstention::NoCandidates));
    }

    #[test]
    fn abstains_when_any_other_family_is_as_strong() {
        let mut best = Fixture::scored(1, "f_best", 1.0, &["src/paginate.js"]);
        best.signals.lexical = Some((0, 1.0));
        let weak = Fixture::scored(2, "f_weak", 0.99, &[]);
        let strong = Fixture::scored(3, "f_strong", 0.97, &["src/paginate.js"]);
        let distant = Fixture::scored(4, "f_distant", 0.9, &["src/paginate.js"]);

        let ambiguous = Decision::Abstain(Abstention::Ambiguous);
        assert_eq!(
            Fixture::gate(&[best.clone(), weak.clone(), strong.clone()]),
            ambiguous
        );
        assert_eq!(Fixture::gate(&[best.clone(), strong]), ambiguous);
        assert!(matches!(
            Fixture::gate(&[best.clone(), weak.clone()]),
            Decision::Inject(_)
        ));
        assert!(matches!(
            Fixture::gate(&[best, weak, distant]),
            Decision::Inject(_)
        ));
    }

    #[test]
    fn the_prompts_task_kind_settles_a_tie() {
        let mut other_kind = Fixture::scored(1, "f_add", 1.0, &["src/paginate.js"]);
        other_kind.signals.lexical = Some((1, 1.0));
        other_kind.signals.same_kind = Some(false);
        let mut same_kind = Fixture::scored(2, "f_fix", 0.98, &["src/paginate.js"]);
        same_kind.signals.lexical = Some((0, 1.0));
        same_kind.signals.same_kind = Some(true);
        let mut unclear = same_kind.clone();
        unclear.signals.same_kind = None;
        let mut rival = other_kind.clone();
        rival.signals.fused = 0.97;
        rival.signals.same_kind = Some(true);

        let Decision::Inject(served) = Fixture::gate(&[other_kind.clone(), same_kind.clone()])
        else {
            panic!("the task of the prompt's kind is served");
        };
        assert_eq!(served.row.procedure.family.as_str(), "f_fix");
        assert_eq!(
            Fixture::gate(&[other_kind, unclear]),
            Decision::Abstain(Abstention::NotConfident)
        );
        assert_eq!(
            Fixture::gate(&[same_kind, rival]),
            Decision::Abstain(Abstention::Ambiguous)
        );
    }

    #[test]
    fn recalls_the_same_despite_a_huge_paste() {
        let mut store = Store::open_in_memory().expect("in-memory store opens");
        let mut procedure = Procedure::example();
        procedure.preconditions.clear();
        store.upsert(&procedure).expect("procedure stores");
        let paste: String = (0..200_000).map(|n| format!("w{n:x} ")).collect();

        let expected = Fixture::outcome(&store, PROMPT);

        assert!(matches!(expected.decision, Decision::Inject(_)));
        assert_eq!(
            Fixture::outcome(&store, &format!("{PROMPT}\n{paste}")),
            expected
        );
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

    #[derive(Debug)]
    struct Learned(Store);

    impl Learned {
        fn with(entities: &[Entity]) -> Self {
            let mut store = Store::open_in_memory().expect("in-memory store opens");
            let mut procedure = Procedure::example();
            procedure.preconditions.clear();
            procedure.trigger.entities = entities.to_vec();
            store.upsert(&procedure).expect("procedure stores");
            Self(store)
        }

        fn exact(&self, prompt: &str) -> Vec<(String, String)> {
            let mut exact = Fixture::outcome(&self.0, prompt)
                .candidates
                .into_iter()
                .next()
                .expect("the procedure is a candidate")
                .signals
                .exact;
            exact.sort_unstable();
            exact
        }

        fn pairs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
            pairs
                .iter()
                .map(|(key, kind)| ((*key).to_owned(), (*kind).to_owned()))
                .collect()
        }
    }

    #[test]
    fn a_file_name_stem_alone_does_not_inject() {
        let learned = Learned::with(&[
            Entity::Path("src/index.js".to_owned()),
            Entity::Command("npm test".to_owned()),
        ]);
        let bare = "Fix the crash in index when the user logs out.";

        assert_eq!(
            Fixture::outcome(&learned.0, bare).decision,
            Decision::Abstain(Abstention::NotConfident)
        );
        assert_eq!(learned.exact(bare), Learned::pairs(&[("index", "stem")]));
        for prompt in [
            "Fix the crash in src/index.js when the user logs out.",
            "Fix the crash in index.js when the user logs out.",
        ] {
            assert!(
                matches!(
                    Fixture::outcome(&learned.0, prompt).decision,
                    Decision::Inject(_)
                ),
                "{prompt}"
            );
        }
    }

    #[test]
    fn a_file_name_stem_counts_as_a_path_only_when_the_prompt_is_on_topic() {
        for (cosine, weight) in [(None, 0.5), (Some(0.1), 0.5), (Some(0.3), 2.0)] {
            let mut signals = Signals {
                cosine,
                ..Signals::default()
            };
            signals.add_exact("index".to_owned(), "stem".to_owned());

            assert_eq!(signals.exact_weight(), weight, "{cosine:?}");
        }
    }

    #[test]
    fn a_file_without_an_extension_keeps_the_full_path_weight() {
        let learned = Learned::with(&[
            Entity::Path("Makefile".to_owned()),
            Entity::Path("Makefile.am".to_owned()),
        ]);
        let prompt = "Fix the Makefile so the build passes.";

        assert_eq!(
            learned.exact(prompt),
            Learned::pairs(&[("makefile", "path")])
        );
        assert!(matches!(
            Fixture::outcome(&learned.0, prompt).decision,
            Decision::Inject(_)
        ));
    }

    #[test]
    fn a_key_indexed_twice_counts_with_its_strongest_kind() {
        let learned = Learned::with(&[
            Entity::Path("src/paginate.js".to_owned()),
            Entity::Symbol("paginate".to_owned()),
        ]);

        assert_eq!(
            learned.exact("Fix paginate for the last page."),
            Learned::pairs(&[("paginate", "symbol")])
        );
        for order in [["stem", "symbol"], ["symbol", "stem"]] {
            let mut signals = Signals::default();
            for kind in order {
                signals.add_exact("paginate".to_owned(), kind.to_owned());
            }
            assert_eq!(signals.exact_weight(), 1.5, "{order:?}");
        }
    }

    #[test]
    fn matches_commands_written_in_the_prompt() {
        let learned = Learned::with(&[
            Entity::Command("npm test".to_owned()),
            Entity::Command("python -m pytest".to_owned()),
        ]);

        assert_eq!(
            learned.exact("Fix it: `npm test` fails after the upgrade."),
            Learned::pairs(&[("npm test", "command")])
        );
        assert_eq!(
            learned.exact("Fix it: python -m pytest fails after the upgrade."),
            Learned::pairs(&[("python m pytest", "command")])
        );
    }

    #[test]
    fn looks_up_adjacent_words_as_phrases() {
        let words: String = (0..2 * MAX_TERMS).map(|n| format!("w{n} ")).collect();

        assert_eq!(
            Recall::phrases("Run `npm test`, then python -m pytest"),
            [
                "run npm",
                "run npm test",
                "npm test",
                "npm test python",
                "test python",
                "test python m",
                "python m",
                "python m pytest",
                "m pytest",
            ]
        );
        assert_eq!(Recall::phrases(&words).len(), MAX_TERMS);
    }

    #[test]
    fn matches_an_error_pasted_into_the_prompt() {
        let learned = Learned::with(&[Entity::ErrorSignature(
            "TypeError: Cannot read properties of undefined (reading 'id')".to_owned(),
        )]);
        let pasted = "The listing page fails to load:\n\n/home/ada/shop/src/listing.js:3\n    const id = page.id;\n\nTypeError: Cannot read properties of undefined (reading 'id')\n    at render (/home/ada/shop/src/listing.js:3:18)";
        let described = "The listing page throws an error when it loads, fix the error.";

        assert_eq!(
            learned.exact(pasted),
            Learned::pairs(&[(
                "typeerror: cannot read properties of undefined (reading 'id')",
                "error"
            )])
        );
        assert!(matches!(
            Fixture::outcome(&learned.0, pasted).decision,
            Decision::Inject(_)
        ));
        assert_eq!(learned.exact(described), Learned::pairs(&[]));
        assert_eq!(
            Fixture::outcome(&learned.0, described).decision,
            Decision::Abstain(Abstention::NotConfident)
        );
    }

    #[test]
    fn finds_each_pasted_error_up_to_a_limit() {
        let build = "cargo build fails:\n   Compiling shop v0.1.0\nwarning: unused variable: `x`\nerror[E0308]: mismatched types\n --> src/cart.rs:2:22\nerror[E0599]: no method named `totl` found for struct `Cart` in the current scope\n --> src/cart.rs:9:7\nerror[E0308]: mismatched types\n --> src/cart.rs:14:9";
        let flood: String = (0..2 * MAX_ERROR_SIGNATURES)
            .map(|n| format!("error: cannot find value `total{n}` in this scope\n"))
            .collect();

        assert_eq!(
            PastedErrors::signatures(build),
            [
                "error[e0308]: mismatched types",
                "error[e0599]: no method named `totl` found for struct `cart` in the current scope",
            ]
        );
        let signatures = PastedErrors::signatures(&flood);
        assert_eq!(signatures.len(), MAX_ERROR_SIGNATURES);
        assert_eq!(
            signatures[MAX_ERROR_SIGNATURES - 1],
            format!(
                "error: cannot find value `total{}` in this scope",
                MAX_ERROR_SIGNATURES - 1
            )
        );
    }

    #[test]
    fn the_prefilter_keeps_every_line_that_can_have_a_signature() {
        let redactor = Redactor::new();
        let outputs = [
            "Traceback (most recent call last):\n  File \"/home/dev/app/cli.py\", line 3, in <module>\n    import yaml\nModuleNotFoundError: No module named 'yaml'",
            "error[E0432]: unresolved import `crate::commands::total`\n --> src/cli.rs:4:5",
            "/bin/sh: line 1: python: command not found",
            "<path>: line N: python: command not found",
            "Error: Cannot find module '/tmp/run-3/catalog/scripts/build.js'",
            "\x1b[31mTypeError: Cannot read properties of undefined (reading 'id')\x1b[0m",
            "error[E0308]: mismatched types\n --> src/main.rs:2:22\n  |\n2 |     let total: u32 = \"3\";\n  |                ---   ^^^ expected `u32`, found `&str`",
            "   Compiling shop v0.1.0 (/home/dev/shop)\nwarning: unused variable: `x`\nerror[E0599]: no method named `totl` found for struct `Cart` in the current scope\nerror: could not compile `shop` (bin \"shop\") due to 1 previous error",
            "error: expected one of `,`, `:`, or `}`, found `{`",
            "error[E0277]: the trait bound `Total: Serialize` is not satisfied",
            "SyntaxError: Unexpected token '}'",
            "Traceback (most recent call last):\n  File \"/home/dev/app/io.py\", line 9, in <module>\n    open(None)\nTypeError: expected str, bytes or os.PathLike object, not NoneType",
            "  File \"/home/dev/app/cli.py\", line 3\n    def main(\n            ^\nSyntaxError: '(' was never closed",
            "Traceback (most recent call last):\n  File \"/home/dev/app/cli.py\", line 3, in <module>\n    raise ConfigMissing\nshop.errors.ConfigMissingError",
            "# example.com/shop\n./main.go:5:2: undefined: totalPrice\n./main.go:9:7: \"fmt\" imported and not used",
            "internal/cart/cart.go:41:12: cannot use price (variable of type float64) as int value in return statement",
            "cannot use price (variable of type float64) as int value in return statement",
            "panic: runtime error: index out of range [5] with length 3\n\ngoroutine 1 [running]:\nmain.main()\n\t/home/dev/shop/main.go:8 +0x1d",
            "panic: runtime error: index out of range [N] with length N",
            "src/cart.ts(3,5): error TS2322: Type 'string' is not assignable to type 'number'.",
            "src/cart.ts:3:5 - error TS2322: Type 'string' is not assignable to type 'number'.",
            "error TS5058: The specified path does not exist: 'tsconfig.app.json'.",
            "/home/dev/shop/index.js:3\nconst id = order.id;\n                 ^\n\nTypeError: Cannot read properties of undefined (reading 'id')\n    at Object.<anonymous> (/home/dev/shop/index.js:3:18)",
            "node:internal/fs/utils:347\n    throw err;\nTypeError [ERR_INVALID_ARG_TYPE]: The \"path\" argument must be of type string. Received undefined",
            "Uncaught ReferenceError: process is not defined",
            "npm ERR! Missing script: \"build\"\nnpm ERR!\nnpm ERR! To see a list of scripts, run:\nnpm ERR!   npm run",
            "npm ERR! code E404\nnpm ERR! 404 Not Found - GET https://registry.npmjs.org/left-padd - Not found",
            "npm err! N not found - get https:<path> - not found",
            "npm error code ERESOLVE\nnpm error ERESOLVE unable to resolve dependency tree",
            "Exception in thread \"main\" java.lang.NullPointerException: Cannot invoke \"String.length()\" because \"name\" is null\n\tat Shop.main(Shop.java:5)",
            "Exception in thread \"main\" java.lang.IllegalStateException\n\tat Shop.main(Shop.java:5)",
            "Shop.java:5: error: cannot find symbol\n    total = prise * 2;\n            ^\n  symbol:   variable prise",
            "cart.c: In function 'main':\ncart.c:3:9: warning: unused variable 'n' [-Wunused-variable]\ncart.c:5:5: error: use of undeclared identifier 'totl'",
            "cart.c:1:10: fatal error: 'shop.h' file not found\n#include \"shop.h\"\n         ^~~~~~~~",
            "fatal error: 'shop.h' file not found",
            "/usr/bin/ld: /tmp/ccq1.o: in function `main':\ncart.c:(.text+0x9): undefined reference to `total'\ncollect2: error: ld returned 1 exit status",
            "cart.c:(.text+<hex>): undefined reference to `total'",
            "shop/cart.py:12: error: Incompatible return value type (got \"str\", expected \"int\")  [return-value]\nFound 1 error in 1 file (checked 3 source files)",
            "/home/ada/shop/src/lib/total.ts(88,13): error TS2322: Type 'string' is not assignable to type 'number'.",
            r"Error: Cannot find module 'C:\Users\grace\shop\scripts\build.js'",
            r"FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\grace\\shop\\cart.toml'",
            r"Error: ENOENT: no such file or directory, open 'd:/clients/grace/cart.json'",
            r"Error: Cannot find module '\\fileserver\grace\build.js'",
            r"SyntaxError: invalid escape sequence '\d'",
            r"C:\Users\grace\shop\src\cart.c:41:5: error: use of undeclared identifier 'totl'",
            "Error: password authentication failed for user \"app\"",
            "src/lib.rs:10:5",
            "Exit code 1\n3 passing, 1 failing",
            "src/cart.rs:3:5: warning: unused variable: `x`",
            "FAILED (failures=1)",
        ];
        let signed: Vec<&str> = outputs
            .iter()
            .flat_map(|output| output.lines())
            .filter(|line| ErrorSignature::of(line, &redactor).is_some())
            .collect();

        assert!(signed.len() > outputs.len() / 2, "{signed:?}");
        for line in signed {
            assert!(PastedErrors::may_be_error(line), "{line}");
        }
    }

    #[test]
    fn the_prefilter_skips_ordinary_prose() {
        for line in [
            PROMPT,
            CRASH_PROMPT,
            "Meet at 10:30 and check the build in src/cart.rs:12 before lunch.",
            "Call total(3, 4) and compare the result with the CSV.",
            "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9",
        ] {
            assert!(!PastedErrors::may_be_error(line), "{line}");
        }
    }

    #[test]
    fn a_prompt_that_only_mentions_an_error_has_no_signature() {
        for prompt in [
            "Fix the error in src/cart.rs",
            "Error",
            "Why does this error happen when I run cargo build?",
            "error handling in the cart is wrong",
            "The TypeError is back, can you look at the listing page?",
            "warning: unused variable: `x`",
        ] {
            assert_eq!(
                PastedErrors::signatures(prompt),
                Vec::<String>::new(),
                "{prompt}"
            );
        }
    }

    #[derive(Debug)]
    struct Checkout {
        root: std::path::PathBuf,
    }

    impl Checkout {
        fn new(name: &str) -> Self {
            let root = std::env::temp_dir().join(format!(
                "trodden-preconditions-{name}-{}",
                std::process::id()
            ));
            fs::create_dir_all(root.join("src")).expect("scratch directory is creatable");
            Self { root }
        }

        fn write(&self, path: &str, contents: &[u8]) -> &Self {
            fs::write(self.root.join(path), contents).expect("scratch file is writable");
            self
        }

        fn failure(&self, condition: Condition) -> Option<String> {
            let mut procedure = Procedure::example();
            procedure.preconditions = vec![condition];
            Preconditions::first_failure(&procedure, &self.root)
        }

        fn mentions(&self, path: &str) -> bool {
            self.failure(Condition::SymbolInFile {
                path: path.to_owned(),
                symbol: "paginate".to_owned(),
            })
            .is_none()
        }
    }

    impl Drop for Checkout {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).expect("scratch directory is removable");
        }
    }

    #[test]
    fn reads_symbols_only_from_files_inside_the_repository() {
        let checkout = Checkout::new("inside");
        checkout.write("src/paginate.js", b"export function paginate() {}\n");
        let outside = checkout
            .root
            .join("src/paginate.js")
            .to_string_lossy()
            .into_owned();

        assert!(checkout.mentions("src/paginate.js"));
        assert!(checkout.mentions("./src/paginate.js"));
        assert!(!checkout.mentions(&outside));
        assert!(!checkout.mentions("src/../src/paginate.js"));
        assert!(!checkout.mentions("src/missing.js"));
        assert!(!checkout.mentions("src"));
    }

    #[test]
    fn reads_only_the_start_of_a_huge_file() {
        let checkout = Checkout::new("huge");
        let mut early = b"paginate\n".to_vec();
        early.resize(4 * 1024 * 1024, b'x');
        let mut late = vec![b'x'; 4 * 1024 * 1024];
        late.extend(b"paginate\n");
        checkout
            .write("src/early.log", &early)
            .write("src/late.log", &late);

        assert!(checkout.mentions("src/early.log"));
        assert!(!checkout.mentions("src/late.log"));
    }

    #[cfg(unix)]
    #[test]
    fn follows_symlinks_only_to_regular_files_inside_the_repository() {
        use std::os::unix::fs::symlink;

        let checkout = Checkout::new("symlinks");
        let elsewhere = Checkout::new("symlinks-elsewhere");
        checkout.write("src/paginate.js", b"export function paginate() {}\n");
        elsewhere.write("src/paginate.js", b"export function paginate() {}\n");
        let link = |target: &Path, name: &str| {
            symlink(target, checkout.root.join(name)).expect("symlink is creatable");
        };
        link(Path::new("paginate.js"), "src/linked.js");
        link(Path::new("/dev/zero"), "src/zero.js");
        link(&elsewhere.root.join("src/paginate.js"), "src/outside.js");

        assert!(checkout.mentions("src/linked.js"));
        assert!(!checkout.mentions("src/zero.js"));
        assert!(!checkout.mentions("src/outside.js"));
    }

    #[cfg(unix)]
    #[test]
    fn does_not_wait_on_a_fifo() {
        let checkout = Checkout::new("fifo");
        let status = std::process::Command::new("mkfifo")
            .arg(checkout.root.join("src/paginate.js"))
            .status()
            .expect("mkfifo runs");
        assert!(status.success());

        assert!(!checkout.mentions("src/paginate.js"));
    }

    #[test]
    fn finds_relative_programs_from_the_repository_root() {
        let checkout = Checkout::new("programs");
        fs::create_dir_all(checkout.root.join(".venv/bin")).expect("venv is creatable");
        checkout.write(".venv/bin/pytest", b"");
        let program = |program: &str| Condition::ProgramOnPath {
            program: program.to_owned(),
        };

        assert_eq!(checkout.failure(program(".venv/bin/pytest")), None);
        assert_eq!(checkout.failure(program("./.venv/bin/pytest")), None);
        assert_eq!(
            checkout.failure(program(".venv/bin/ruff")),
            Some(".venv/bin/ruff is not on PATH".to_owned())
        );
        assert_eq!(
            checkout.failure(program(".venv/bin")),
            Some(".venv/bin is not on PATH".to_owned())
        );
    }
}
