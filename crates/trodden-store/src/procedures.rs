use std::{borrow::Cow, fmt::Write};

use anyhow::{Context, Result};
use jiff::Timestamp;
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};
use sha2::{Digest, Sha256};
use trodden_core::{
    FamilyId, Procedure, ProcedureId, TaskKind,
    procedure::{Entity, Lifecycle, Scope, Slot},
};

use crate::{Store, Terms};

pub(crate) const LEXICAL_POSTINGS_BUDGET: i64 = 2_000;

const MAX_EXAMPLES: usize = 8;

const MAX_AVOID: usize = 3;

type FamilyRow = (i64, u32, String, String, String);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Upsert {
    Created { rowid: i64 },
    Refreshed { rowid: i64 },
    Generalized { rowid: i64 },
    Revised { rowid: i64 },
}

impl Upsert {
    pub fn rowid(self) -> i64 {
        match self {
            Self::Created { rowid }
            | Self::Refreshed { rowid }
            | Self::Generalized { rowid }
            | Self::Revised { rowid } => rowid,
        }
    }
}

#[derive(Debug)]
struct Placement<'a> {
    candidate: Cow<'a, Procedure>,
    family: Vec<FamilyRow>,
}

#[derive(Debug)]
struct Absorber {
    rowid: i64,
    stored: Procedure,
    state: Lifecycle,
    generalized: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProcedureRow {
    pub rowid: i64,
    pub procedure: Procedure,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityHit {
    pub rowid: i64,
    pub key: String,
    pub kind: String,
}

impl EntityHit {
    pub fn is_stem_of(&self, procedure: &Procedure) -> bool {
        self.kind == "path"
            && !procedure
                .trigger
                .entities
                .iter()
                .any(|entity| match entity {
                    Entity::Path(path) => Store::file_keys(path).contains(&self.key),
                    _ => false,
                })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalHit {
    pub rowid: i64,
    pub score: f64,
}

impl Store {
    pub fn upsert(&mut self, candidate: &Procedure) -> Result<Upsert> {
        let tx = self.storing()?;
        let placement = Self::place(&tx, candidate)?;
        let outcome = Self::save(&tx, &placement.family, &placement.candidate)?;
        tx.commit().context("commit a procedure")?;
        Ok(outcome)
    }

    fn storing(&mut self) -> Result<Transaction<'_>> {
        self.conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start storing a procedure")
    }

    fn place<'a>(tx: &Transaction<'_>, candidate: &'a Procedure) -> Result<Placement<'a>> {
        let family = Self::family(tx, &candidate.family)?;
        match candidate.trigger.kind() {
            Some(kind) if Self::kind_of(&family)?.is_some_and(|known| known != kind) => {
                let sibling = Self::sibling(candidate, kind);
                Ok(Placement {
                    family: Self::family(tx, &sibling.family)?,
                    candidate: Cow::Owned(sibling),
                })
            }
            _ => Ok(Placement {
                candidate: Cow::Borrowed(candidate),
                family,
            }),
        }
    }

    fn kind_of(family: &[FamilyRow]) -> Result<Option<TaskKind>> {
        for (_, _, _, document, _) in family {
            let stored: Procedure =
                serde_json::from_str(document).context("parse a stored procedure")?;
            if let Some(kind) = stored.trigger.kind() {
                return Ok(Some(kind));
            }
        }
        Ok(None)
    }

    fn sibling(candidate: &Procedure, kind: TaskKind) -> Procedure {
        let digest = Sha256::digest(format!("{}\n{}", candidate.family.as_str(), kind.as_str()));
        let hash = digest.iter().take(6).fold(String::new(), |mut hex, byte| {
            write!(hex, "{byte:02x}").expect("writing to a String cannot fail");
            hex
        });
        let mut sibling = candidate.clone();
        sibling.id = ProcedureId::new(format!("p_{hash}"));
        sibling.family = FamilyId::new(format!("f_{hash}"));
        sibling
    }

    fn family(tx: &Transaction<'_>, family: &FamilyId) -> Result<Vec<FamilyRow>> {
        let mut statement = tx
            .prepare_cached("SELECT rowid, revision, signature, document, state FROM procedures WHERE family = ?1 ORDER BY rowid")
            .context("prepare the family lookup")?;
        statement
            .query_map([family.as_str()], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                ))
            })
            .context("look up the procedure family")?
            .collect::<Result<_, _>>()
            .context("read the procedure family")
    }

    fn save(tx: &Transaction<'_>, existing: &[FamilyRow], candidate: &Procedure) -> Result<Upsert> {
        let signature = Self::signature(candidate)?;
        let outcome = if let Some(mut absorber) = Self::absorber(existing, candidate, &signature)? {
            let stored = &mut absorber.stored;
            Self::merge(stored, candidate);
            stored.state = Self::refreshed_state(absorber.state, absorber.rowid, existing);
            tx.execute(
                "UPDATE procedures SET state = ?1, document = ?2, signature = ?3, updated_at = ?4, used_at = ?5
                 WHERE rowid = ?6",
                params![
                    Self::state_name(stored.state),
                    Self::document(stored)?,
                    Self::signature(stored)?,
                    stored.provenance.updated_at.to_string(),
                    Timestamp::now().to_string(),
                    absorber.rowid
                ],
            )
            .context("refresh a procedure revision")?;
            if stored.state == Lifecycle::Active {
                Self::reindex(tx, stored.id.as_str())?;
            }
            let rowid = absorber.rowid;
            if absorber.generalized {
                Upsert::Generalized { rowid }
            } else {
                Upsert::Refreshed { rowid }
            }
        } else {
            let mut stored = candidate.clone();
            stored.revision = existing.iter().map(|row| row.1).max().unwrap_or(0) + 1;
            let states: Vec<Lifecycle> = existing
                .iter()
                .map(|row| Self::parse_state(&row.4))
                .collect();
            stored.state = if states
                .iter()
                .any(|state| matches!(state, Lifecycle::Active | Lifecycle::Stale))
            {
                Lifecycle::Candidate
            } else if !states.is_empty() && states.iter().all(|state| *state == Lifecycle::Retired)
            {
                Lifecycle::Retired
            } else {
                Lifecycle::Active
            };
            if let Some(id) = existing
                .first()
                .and_then(|row| serde_json::from_str::<Procedure>(&row.3).ok())
            {
                stored.id = id.id;
            }
            tx.execute(
                "INSERT INTO procedures (id, revision, family, repo, state, title, signature, document, updated_at, used_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    stored.id.as_str(),
                    stored.revision,
                    stored.family.as_str(),
                    Self::repo_of(&stored),
                    Self::state_name(stored.state),
                    stored.title,
                    signature,
                    Self::document(&stored)?,
                    stored.provenance.updated_at.to_string(),
                    Timestamp::now().to_string(),
                ],
            )
            .context("insert a procedure revision")?;
            let rowid = tx.last_insert_rowid();
            if stored.state == Lifecycle::Active {
                Self::reindex(tx, stored.id.as_str())?;
                Upsert::Created { rowid }
            } else {
                Upsert::Revised { rowid }
            }
        };
        Ok(outcome)
    }

    fn absorber(
        family: &[FamilyRow],
        candidate: &Procedure,
        signature: &str,
    ) -> Result<Option<Absorber>> {
        let parse = |document: &str| -> Result<Procedure> {
            serde_json::from_str(document).context("parse a stored procedure")
        };
        if let Some((rowid, _, _, document, state)) = family.iter().find(|row| row.2 == signature) {
            return Ok(Some(Absorber {
                rowid: *rowid,
                stored: parse(document)?,
                state: Self::parse_state(state),
                generalized: false,
            }));
        }
        for (rowid, _, _, document, state) in family {
            let state = Self::parse_state(state);
            if !matches!(
                state,
                Lifecycle::Active
                    | Lifecycle::Stale
                    | Lifecycle::Candidate
                    | Lifecycle::Quarantined
                    | Lifecycle::Retired
            ) {
                continue;
            }
            if let Some(stored) = parse(document)?.anti_unify(candidate) {
                return Ok(Some(Absorber {
                    rowid: *rowid,
                    stored,
                    state,
                    generalized: true,
                }));
            }
        }
        Ok(None)
    }

    fn refreshed_state(state: Lifecycle, rowid: i64, family: &[FamilyRow]) -> Lifecycle {
        match state {
            Lifecycle::Stale => Lifecycle::Active,
            Lifecycle::Archived
                if family.iter().any(|row| {
                    row.0 != rowid
                        && matches!(
                            Self::parse_state(&row.4),
                            Lifecycle::Active | Lifecycle::Stale
                        )
                }) =>
            {
                Lifecycle::Candidate
            }
            Lifecycle::Archived => Lifecycle::Active,
            other => other,
        }
    }

    fn merge(stored: &mut Procedure, candidate: &Procedure) {
        for source in &candidate.provenance.sources {
            if !stored
                .provenance
                .sources
                .iter()
                .any(|known| known.session == source.session)
            {
                stored.provenance.sources.push(source.clone());
            }
        }
        stored.provenance.updated_at = stored
            .provenance
            .updated_at
            .max(candidate.provenance.updated_at);

        let prompt = &candidate.trigger.text;
        if *prompt != stored.trigger.text
            && !stored.trigger.examples.contains(prompt)
            && stored.trigger.examples.len() < MAX_EXAMPLES
        {
            stored.trigger.examples.push(prompt.clone());
        }
        for entity in &candidate.trigger.entities {
            if !stored.trigger.entities.contains(entity) {
                stored.trigger.entities.push(entity.clone());
            }
        }
        for slot in &candidate.slots {
            if let Some(known) = stored
                .slots
                .iter_mut()
                .find(|known| known.name == slot.name)
            {
                for example in &slot.examples {
                    if !known.examples.contains(example)
                        && known.examples.len() < Slot::MAX_EXAMPLES
                    {
                        known.examples.push(example.clone());
                    }
                }
            }
        }
        for line in &candidate.avoid {
            if !stored.avoid.contains(line) && stored.avoid.len() < MAX_AVOID {
                stored.avoid.push(line.clone());
            }
        }
    }

    pub(crate) fn reindex(tx: &Transaction<'_>, procedure: &str) -> Result<()> {
        tx.execute(
            "DELETE FROM procedures_fts WHERE rowid IN (SELECT rowid FROM procedures WHERE id = ?1)",
            [procedure],
        )
        .context("remove a procedure from the text index")?;
        tx.execute(
            "DELETE FROM entities WHERE procedure IN (SELECT rowid FROM procedures WHERE id = ?1)",
            [procedure],
        )
        .context("remove a procedure's entities")?;
        let incumbent: Option<(i64, String)> = tx
            .query_row(
                "SELECT rowid, document FROM procedures
                 WHERE id = ?1 AND state IN ('active', 'stale') ORDER BY revision DESC LIMIT 1",
                [procedure],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .context("find a procedure's incumbent")?;
        if let Some((rowid, document)) = incumbent {
            let stored: Procedure =
                serde_json::from_str(&document).context("parse a stored procedure")?;
            Self::index(tx, rowid, &stored)?;
        }
        Ok(())
    }

    fn index(tx: &Transaction<'_>, rowid: i64, procedure: &Procedure) -> Result<()> {
        let mut body = vec![procedure.trigger.text.clone()];
        body.extend(procedure.trigger.examples.iter().cloned());
        for step in &procedure.steps {
            body.extend(step.target.iter().cloned());
            body.extend(step.command.iter().cloned());
            body.extend(step.symbols.iter().cloned());
        }
        tx.execute(
            "INSERT INTO procedures_fts (rowid, title, body) VALUES (?1, ?2, ?3)",
            params![
                rowid,
                Terms::normalize(&procedure.title),
                Terms::normalize(&body.join(" "))
            ],
        )
        .context("index a procedure's text")?;

        let mut statement = tx
            .prepare_cached("INSERT INTO entities (key, kind, procedure) VALUES (?1, ?2, ?3)")
            .context("prepare the entity insert")?;
        for (key, kind) in procedure
            .trigger
            .entities
            .iter()
            .flat_map(Self::entity_keys)
        {
            statement
                .execute(params![key, kind, rowid])
                .context("index a procedure entity")?;
        }
        Ok(())
    }

    fn entity_keys(entity: &Entity) -> Vec<(String, &'static str)> {
        match entity {
            Entity::Path(path) => {
                let keys = Self::file_keys(path);
                let name = keys.last().expect("a path has a file name key");
                let stem = name
                    .split('.')
                    .next()
                    .filter(|stem| stem != name && stem.len() > 2)
                    .map(|stem| (stem.to_owned(), "path"));
                keys.into_iter()
                    .map(|key| (key, "path"))
                    .chain(stem)
                    .collect()
            }
            Entity::Symbol(symbol) => vec![(symbol.to_lowercase(), "symbol")],
            Entity::Command(command) => {
                let words = Terms::normalize(command);
                (!words.is_empty())
                    .then_some((words, "command"))
                    .into_iter()
                    .collect()
            }
            Entity::ErrorSignature(signature) => vec![(signature.to_lowercase(), "error")],
            Entity::Package(package) => vec![(package.to_lowercase(), "package")],
            _ => Vec::new(),
        }
    }

    fn file_keys(path: &str) -> Vec<String> {
        let path = path.to_lowercase();
        let name = path.rsplit('/').next().unwrap_or(&path).to_owned();
        if name == path {
            vec![path]
        } else {
            vec![path, name]
        }
    }

    pub fn entity_hits(&self, keys: &[String], repo: &str) -> Result<Vec<EntityHit>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders = vec!["?"; keys.len()].join(", ");
        let sql = format!(
            "SELECT e.procedure, e.key, e.kind FROM entities e
             CROSS JOIN procedures p ON p.rowid = e.procedure
             WHERE e.key IN ({placeholders}) AND p.state IN ('active', 'stale') AND (p.repo = ? OR p.repo = '')"
        );
        let mut statement = self
            .conn
            .prepare(&sql)
            .context("prepare the entity lookup")?;
        let values = keys.iter().map(String::as_str).chain([repo]);
        statement
            .query_map(rusqlite::params_from_iter(values), |row| {
                Ok(EntityHit {
                    rowid: row.get(0)?,
                    key: row.get(1)?,
                    kind: row.get(2)?,
                })
            })
            .context("look up entities")?
            .collect::<Result<_, _>>()
            .context("read entity matches")
    }

    pub fn lexical_hits(
        &self,
        terms: &[String],
        repo: &str,
        limit: usize,
    ) -> Result<Vec<LexicalHit>> {
        let mut frequencies: Vec<(&str, i64)> = Vec::new();
        {
            let mut statement = self
                .conn
                .prepare("SELECT doc FROM procedures_vocab WHERE term = ?1")
                .context("prepare the term frequency lookup")?;
            for term in terms {
                if frequencies.iter().any(|(known, _)| known == term) {
                    continue;
                }
                let documents: Option<i64> = statement
                    .query_row([term], |row| row.get(0))
                    .optional()
                    .context("look up a term frequency")?;
                if let Some(documents) = documents {
                    frequencies.push((term, documents));
                }
            }
        }
        frequencies.sort_by_key(|(_, documents)| *documents);
        let mut postings = 0;
        let selected: Vec<&str> = frequencies
            .iter()
            .take_while(|(_, documents)| {
                let within = postings == 0 || postings + documents <= LEXICAL_POSTINGS_BUDGET;
                postings += documents;
                within
            })
            .map(|(term, _)| *term)
            .collect();
        if selected.is_empty() {
            return Ok(Vec::new());
        }

        let query = selected
            .iter()
            .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
            .collect::<Vec<_>>()
            .join(" OR ");
        let mut statement = self
            .conn
            .prepare(
                "SELECT procedures_fts.rowid, -procedures_fts.rank FROM procedures_fts
                 CROSS JOIN procedures p ON p.rowid = procedures_fts.rowid
                 WHERE procedures_fts MATCH ?1 AND p.state IN ('active', 'stale') AND (p.repo = ?2 OR p.repo = '')
                 ORDER BY procedures_fts.rank LIMIT ?3",
            )
            .context("prepare the lexical search")?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        statement
            .query_map(params![query, repo, limit], |row| {
                Ok(LexicalHit {
                    rowid: row.get(0)?,
                    score: row.get(1)?,
                })
            })
            .context("search procedure text")?
            .collect::<Result<_, _>>()
            .context("read lexical matches")
    }

    pub fn is_recallable(&self, rowid: i64, repo: &str) -> Result<bool> {
        self.conn
            .prepare_cached("SELECT 1 FROM procedures WHERE rowid = ?1 AND state IN ('active', 'stale') AND (repo = ?2 OR repo = '')")
            .context("prepare the scope check")?
            .exists(params![rowid, repo])
            .context("check a match's scope")
    }

    pub fn recallable_procedure(&self, rowid: i64, repo: &str) -> Result<Option<ProcedureRow>> {
        if !self.is_recallable(rowid, repo)? {
            return Ok(None);
        }
        self.procedure(rowid)
    }

    pub fn procedure(&self, rowid: i64) -> Result<Option<ProcedureRow>> {
        let document: Option<String> = self
            .conn
            .query_row(
                "SELECT document FROM procedures WHERE rowid = ?1",
                [rowid],
                |row| row.get(0),
            )
            .optional()
            .context("read a procedure")?;
        document
            .map(|document| {
                let procedure =
                    serde_json::from_str(&document).context("parse a stored procedure")?;
                Ok(ProcedureRow { rowid, procedure })
            })
            .transpose()
    }

    pub fn recallable_embeddings(&self) -> Result<Vec<(i64, String, Vec<u8>)>> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT p.rowid, p.repo, e.embedding FROM embeddings e
                 JOIN procedures p ON p.rowid = e.procedure
                 WHERE p.state IN ('active', 'stale') ORDER BY p.rowid, e.rowid",
            )
            .context("prepare the embedding scan")?;
        statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .context("scan embeddings")?
            .collect::<Result<_, _>>()
            .context("read embeddings")
    }

    pub fn set_embeddings(&mut self, rowid: i64, embeddings: &[Vec<u8>]) -> Result<()> {
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start storing embeddings")?;
        tx.execute("DELETE FROM embeddings WHERE procedure = ?1", [rowid])
            .context("remove old embeddings")?;
        for embedding in embeddings {
            tx.execute(
                "INSERT INTO embeddings (procedure, embedding) VALUES (?1, ?2)",
                params![rowid, embedding],
            )
            .context("store an embedding")?;
        }
        tx.commit().context("commit embeddings")
    }

    pub fn list(&self, repo: Option<&str>, all_revisions: bool) -> Result<Vec<ProcedureRow>> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT rowid, document FROM procedures
                 WHERE (?1 IS NULL OR repo = ?1) AND (?2 OR state IN ('active', 'stale'))
                 ORDER BY updated_at DESC, rowid DESC",
            )
            .context("prepare the procedure listing")?;
        let rows = statement
            .query_map(params![repo, all_revisions], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .context("list procedures")?;
        rows.map(|row| {
            let (rowid, document) = row.context("read a listed procedure")?;
            let procedure = serde_json::from_str(&document).context("parse a stored procedure")?;
            Ok(ProcedureRow { rowid, procedure })
        })
        .collect()
    }

    pub fn revisions(&self, id: &str) -> Result<Vec<ProcedureRow>> {
        let mut statement = self
            .conn
            .prepare("SELECT rowid, document FROM procedures WHERE id = ?1 ORDER BY revision")
            .context("prepare the revision lookup")?;
        let rows = statement
            .query_map([id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .context("look up revisions")?;
        rows.map(|row| {
            let (rowid, document) = row.context("read a revision")?;
            let procedure = serde_json::from_str(&document).context("parse a stored procedure")?;
            Ok(ProcedureRow { rowid, procedure })
        })
        .collect()
    }

    pub fn forget(&mut self, target: Forget<'_>) -> Result<usize> {
        let (condition, injected, value) = match target {
            Forget::Procedure(id) => ("id = ?1", "procedure = ?1", Some(id)),
            Forget::Repo(repo) => (
                "repo = ?1",
                "procedure IN (SELECT id FROM procedures WHERE repo = ?1)",
                Some(repo),
            ),
            Forget::All => ("?1 IS NULL", "?1 IS NULL", None),
        };
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start forgetting")?;
        tx.execute(&format!("DELETE FROM injections WHERE {injected}"), [value])
            .context("delete the injections of forgotten procedures")?;
        tx.execute(
            &format!("DELETE FROM extractions WHERE procedure IN (SELECT rowid FROM procedures WHERE {condition})"),
            [value],
        )
        .context("delete the tasks forgotten procedures were learned from")?;
        tx.execute(
            &format!("DELETE FROM procedures_fts WHERE rowid IN (SELECT rowid FROM procedures WHERE {condition})"),
            [value],
        )
        .context("remove forgotten procedures from the text index")?;
        let deleted = tx
            .execute(
                &format!("DELETE FROM procedures WHERE {condition}"),
                [value],
            )
            .context("delete forgotten procedures")?;
        if matches!(target, Forget::All) {
            tx.execute_batch("DELETE FROM extractions; DELETE FROM sessions; DELETE FROM repos;")
                .context("delete ingest history")?;
            tx.execute(
                "INSERT INTO settings (key, value) VALUES ('forgotten_before', ?1)
                 ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                [Timestamp::now().to_string()],
            )
            .context("remember when everything was forgotten")?;
        }
        tx.commit().context("commit forgetting")?;
        Ok(deleted)
    }

    fn signature(procedure: &Procedure) -> Result<String> {
        let steps: Vec<_> = procedure
            .steps
            .iter()
            .map(|step| (step.kind, &step.target, &step.command))
            .collect();
        serde_json::to_string(&steps).context("serialize a procedure signature")
    }

    pub(crate) fn document(procedure: &Procedure) -> Result<String> {
        serde_json::to_string(procedure).context("serialize a procedure")
    }

    fn repo_of(procedure: &Procedure) -> &str {
        match &procedure.scope {
            Scope::Repo { repo } | Scope::Paths { repo, .. } => repo.as_str(),
            _ => "",
        }
    }

    pub(crate) fn state_name(state: Lifecycle) -> &'static str {
        match state {
            Lifecycle::Active => "active",
            Lifecycle::Stale => "stale",
            Lifecycle::Archived => "archived",
            Lifecycle::Quarantined => "quarantined",
            Lifecycle::Retired => "retired",
            _ => "candidate",
        }
    }

    pub(crate) fn parse_state(name: &str) -> Lifecycle {
        match name {
            "active" => Lifecycle::Active,
            "stale" => Lifecycle::Stale,
            "archived" => Lifecycle::Archived,
            "quarantined" => Lifecycle::Quarantined,
            "retired" => Lifecycle::Retired,
            _ => Lifecycle::Candidate,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Forget<'a> {
    Procedure(&'a str),
    Repo(&'a str),
    All,
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use rusqlite::ErrorCode;
    use trodden_core::{FamilyId, ProcedureId, RepoId, SessionId, procedure::StepKind};
    use trodden_learn::Evidence;

    use super::*;
    use crate::{Cue, ExtractionRecord, FamilyEvidence, Injection, Patience, Stats};

    const REPO: &str = "4b1d0c9e8f7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c";

    const SESSION: &str = "0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90";

    const ELSEWHERE: &str = "9e8d7c6b5a4f3e2d1c0b9a8f7e6d5c4b3a2f1e0d";

    #[derive(Debug)]
    struct Tasks(Store);

    impl Tasks {
        const FIX: &str = "Page 2 repeats the last product from page 1. Fix it.";
        const ADD: &str = "Add a sort parameter to paginate";
        const UNCLEAR: &str = "perPage values above 100 should be capped at 100.";

        fn new() -> Self {
            Self(Store::open_in_memory().expect("store opens"))
        }

        fn learn(&mut self, prompt: &str) -> Upsert {
            let mut procedure = Procedure::example();
            procedure.title = prompt.to_owned();
            procedure.trigger.text = prompt.to_owned();
            self.0.upsert(&procedure).expect("stored")
        }

        fn procedures(&self) -> Vec<(String, String)> {
            self.0
                .conn
                .prepare("SELECT id, title FROM procedures ORDER BY rowid")
                .expect("procedure query prepares")
                .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .expect("procedures read")
                .collect::<Result<_, _>>()
                .expect("procedure rows read")
        }
    }

    #[derive(Debug)]
    struct Zebras(Store);

    impl Zebras {
        fn new() -> Self {
            Self(Store::open_in_memory().expect("store opens"))
        }

        fn add(&mut self, name: &str, scope: Scope, title: &str) -> &mut Self {
            let mut procedure = Procedure::example();
            procedure.id = ProcedureId::new(format!("p_{name}"));
            procedure.family = FamilyId::new(format!("f_{name}"));
            procedure.scope = scope;
            procedure.title = title.to_owned();
            self.0.upsert(&procedure).expect("stored");
            self
        }

        fn in_repo(repo: &str) -> Scope {
            Scope::Repo {
                repo: RepoId::new(repo),
            }
        }

        fn crowd(&mut self, count: usize) -> &mut Self {
            for n in 0..count {
                self.add(
                    &format!("crowd{n}"),
                    Self::in_repo(ELSEWHERE),
                    "Zebra zebra",
                );
            }
            self
        }

        fn found(&self, limit: usize) -> Vec<String> {
            self.0
                .lexical_hits(&["zebra".to_owned()], REPO, limit)
                .expect("text searched")
                .into_iter()
                .map(|hit| {
                    self.0
                        .procedure(hit.rowid)
                        .expect("procedure reads")
                        .expect("hit exists")
                        .procedure
                        .id
                        .as_str()
                        .to_owned()
                })
                .collect()
        }
    }

    #[derive(Debug)]
    struct Contention {
        dir: PathBuf,
    }

    impl Contention {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("trodden-contention-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch directory is writable");
            Self { dir }
        }

        fn open(&self, patience: Patience) -> Store {
            Store::open(&self.dir.join("trodden.db"), patience).expect("store opens")
        }

        fn injection() -> Injection {
            Injection {
                session: SESSION.to_owned(),
                procedure: "p_0c4e2a9b".to_owned(),
                revision: 1,
                holdout: false,
                cue: Cue::Prompt,
                at: "2026-09-21T14:03:00Z".parse().expect("timestamp is valid"),
            }
        }

        fn is_busy(error: &anyhow::Error) -> bool {
            error
                .downcast_ref::<rusqlite::Error>()
                .and_then(rusqlite::Error::sqlite_error_code)
                == Some(ErrorCode::DatabaseBusy)
        }
    }

    impl Drop for Contention {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[derive(Debug)]
    struct Example;

    impl Example {
        fn relearned(state: Lifecycle, check: &str) -> (Upsert, Store) {
            let mut store = Store::open_in_memory().expect("store opens");
            store.upsert(&Procedure::example()).expect("stored");
            store
                .set_states("p_7f3a91c2", &[(1, state)])
                .expect("state changes");
            let mut later = Procedure::example();
            later.provenance.sources[0].session =
                SessionId::new("9d3e7a10-5b2c-4e8f-a1d6-0c7b9e2f4a58");
            later
                .steps
                .iter_mut()
                .rfind(|step| step.kind == StepKind::Verify)
                .expect("the example has a check")
                .command = Some(check.to_owned());
            let upsert = store.upsert(&later).expect("stored");
            (upsert, store)
        }

        fn states(store: &Store) -> Vec<(u32, Lifecycle)> {
            store
                .revisions("p_7f3a91c2")
                .expect("revisions read")
                .iter()
                .map(|row| (row.procedure.revision, row.procedure.state))
                .collect()
        }

        fn recallable(store: &Store) -> bool {
            !store
                .entity_hits(&["paginate".to_owned()], REPO)
                .expect("entities read")
                .is_empty()
        }

        fn learned_and_injected(store: &mut Store) {
            let rowid = store.upsert(&Procedure::example()).expect("stored").rowid();
            for (first_seq, procedure, rejection) in [
                (0, Some(rowid), None),
                (8, None, Some("no files were changed")),
            ] {
                store
                    .record_extraction(&ExtractionRecord {
                        session: SESSION.to_owned(),
                        first_seq,
                        summary: "Page 2 repeats the last product".to_owned(),
                        procedure,
                        rejection: rejection.map(str::to_owned),
                        outcome: Some("succeeded".to_owned()),
                        tool_calls: Some(6),
                        span: None,
                        at: "2026-09-21T14:02:44Z".to_owned(),
                    })
                    .expect("extraction recorded");
            }
            for procedure in ["p_7f3a91c2", "p_0c4e2a9b"] {
                store
                    .record_injection(&Injection {
                        session: SESSION.to_owned(),
                        procedure: procedure.to_owned(),
                        revision: 1,
                        holdout: false,
                        cue: Cue::Prompt,
                        at: "2026-09-21T14:03:00Z".parse().expect("timestamp is valid"),
                    })
                    .expect("injection recorded");
            }
            let injections = store.injections(None).expect("injections read");
            for record in &injections {
                store
                    .settle_injection(record, 8, "failed")
                    .expect("injection settled");
            }
        }

        fn injected(store: &Store) -> Vec<String> {
            store
                .injections(None)
                .expect("injections read")
                .into_iter()
                .map(|record| record.injection.procedure)
                .collect()
        }
    }

    #[test]
    fn forgetting_a_procedure_deletes_its_injections_and_learned_tasks() {
        for target in [Forget::Procedure("p_7f3a91c2"), Forget::Repo(REPO)] {
            let mut store = Store::open_in_memory().expect("store opens");
            Example::learned_and_injected(&mut store);

            let deleted = store.forget(target).expect("forgotten");

            assert_eq!(deleted, 1, "{target:?}");
            assert_eq!(Example::injected(&store), ["p_0c4e2a9b"], "{target:?}");
            let stats = store.stats().expect("stats read");
            assert_eq!((stats.rejections, stats.injections), (1, 1), "{target:?}");
            assert_eq!(
                store.rejections(10).expect("rejections read").len(),
                1,
                "{target:?}"
            );
        }
    }

    #[test]
    fn a_relearned_procedure_starts_without_the_forgotten_evidence() {
        let mut store = Store::open_in_memory().expect("store opens");
        Example::learned_and_injected(&mut store);
        assert_eq!(
            store
                .family_evidence("p_7f3a91c2")
                .expect("evidence read")
                .injected(),
            Evidence::new(0, 1)
        );

        store
            .forget(Forget::Procedure("p_7f3a91c2"))
            .expect("forgotten");
        store.upsert(&Procedure::example()).expect("relearned");

        assert_eq!(Example::states(&store), [(1, Lifecycle::Active)]);
        assert_eq!(
            store.family_evidence("p_7f3a91c2").expect("evidence read"),
            FamilyEvidence {
                revisions: vec![(1, Lifecycle::Active, Evidence::default())],
                holdout: Evidence::default(),
            }
        );
        assert!(
            store
                .outcome_summaries()
                .expect("summaries read")
                .iter()
                .all(|summary| summary.procedure != "p_7f3a91c2")
        );
    }

    #[test]
    fn forgetting_an_unknown_procedure_deletes_nothing_else() {
        let mut store = Store::open_in_memory().expect("store opens");
        Example::learned_and_injected(&mut store);

        let deleted = store
            .forget(Forget::Procedure("p_doesnotexist"))
            .expect("forgetting is harmless");

        assert_eq!(deleted, 0);
        assert_eq!(Example::injected(&store), ["p_7f3a91c2", "p_0c4e2a9b"]);
        assert_eq!(Example::states(&store), [(1, Lifecycle::Active)]);
    }

    #[test]
    fn forgetting_everything_also_forgets_repositories() {
        let mut store = Store::open_in_memory().expect("store opens");
        Example::learned_and_injected(&mut store);
        store
            .remember_repo("/home/dev/shop", REPO)
            .expect("repository remembered");

        let deleted = store.forget(Forget::All).expect("forgotten");

        assert_eq!(deleted, 1);
        assert!(Example::injected(&store).is_empty());
        assert_eq!(
            store.repo_for_root("/home/dev/shop").expect("repos read"),
            None
        );
        assert_eq!(
            store.stats().expect("stats read"),
            Stats {
                procedures: 0,
                revisions: 0,
                sessions: 0,
                rejections: 0,
                injections: 0,
            }
        );
    }

    #[test]
    fn forgetting_everything_remembers_when() {
        let mut store = Store::open_in_memory().expect("store opens");
        assert_eq!(store.forgotten_before().expect("setting reads"), None);
        let before = Timestamp::now();

        store.forget(Forget::All).expect("forgotten");

        let forgotten = store
            .forgotten_before()
            .expect("setting reads")
            .expect("the forget time is recorded");
        assert!(before <= forgotten && forgotten <= Timestamp::now());
        store.forget(Forget::Repo(REPO)).expect("forgotten again");
        assert_eq!(
            store.forgotten_before().expect("setting reads"),
            Some(forgotten)
        );
    }

    #[test]
    fn relearning_a_quarantined_procedure_keeps_it_quarantined() {
        for (check, expected) in [
            ("npm test", Upsert::Refreshed { rowid: 1 }),
            ("npm check", Upsert::Generalized { rowid: 1 }),
        ] {
            let (upsert, store) = Example::relearned(Lifecycle::Quarantined, check);

            assert_eq!(upsert, expected, "`{check}`");
            assert_eq!(
                Example::states(&store),
                [(1, Lifecycle::Quarantined)],
                "`{check}`"
            );
            assert!(!Example::recallable(&store), "`{check}`");
        }
    }

    #[test]
    fn relearning_a_retired_procedure_keeps_it_retired() {
        for (check, expected) in [
            ("npm test", Upsert::Refreshed { rowid: 1 }),
            ("npm check", Upsert::Generalized { rowid: 1 }),
        ] {
            let (upsert, store) = Example::relearned(Lifecycle::Retired, check);

            assert_eq!(upsert, expected, "`{check}`");
            assert_eq!(
                Example::states(&store),
                [(1, Lifecycle::Retired)],
                "`{check}`"
            );
            assert!(!Example::recallable(&store), "`{check}`");
        }
    }

    #[test]
    fn relearning_an_archived_procedure_brings_it_back_into_recall() {
        let (upsert, store) = Example::relearned(Lifecycle::Archived, "npm test");

        assert_eq!(upsert, Upsert::Refreshed { rowid: 1 });
        assert_eq!(Example::states(&store), [(1, Lifecycle::Active)]);
        assert!(Example::recallable(&store));
    }

    #[test]
    fn relearning_an_archived_procedure_beside_an_active_one_makes_it_a_candidate() {
        let (_, mut store) = Example::relearned(Lifecycle::Archived, "npm check");
        assert_eq!(
            Example::states(&store),
            [(1, Lifecycle::Archived), (2, Lifecycle::Active)]
        );

        let upsert = store.upsert(&Procedure::example()).expect("stored");

        assert_eq!(upsert, Upsert::Refreshed { rowid: 1 });
        assert_eq!(
            Example::states(&store),
            [(1, Lifecycle::Candidate), (2, Lifecycle::Active)]
        );
        assert!(Example::recallable(&store));
    }

    #[test]
    fn storing_a_procedure_survives_a_hook_writing_between_its_read_and_write() {
        let scratch = Contention::new("upsert");
        let mut ingest = scratch.open(Patience::Batch);
        let hook = scratch.open(Patience::Interactive);
        let candidate = Procedure::example();

        let tx = ingest.storing().expect("transaction starts");
        let placement = Store::place(&tx, &candidate).expect("family reads");
        let interleaved = hook.record_injection(&Contention::injection());
        let stored =
            Store::save(&tx, &placement.family, &placement.candidate).expect("procedure is stored");
        tx.commit().expect("procedure commits");

        let refusal = interleaved.expect_err("the hook waits for the procedure to be stored");
        assert!(Contention::is_busy(&refusal), "{refusal:?}");
        hook.record_injection(&Contention::injection())
            .expect("injection recorded once the procedure is stored");
        assert!(matches!(stored, Upsert::Created { .. }));
        assert_eq!(Example::injected(&ingest), ["p_0c4e2a9b"]);
        assert!(Example::recallable(&ingest));
    }

    #[test]
    fn relearning_a_stale_procedure_makes_it_active_again() {
        for (check, expected) in [
            ("npm test", Upsert::Refreshed { rowid: 1 }),
            ("npm check", Upsert::Generalized { rowid: 1 }),
        ] {
            let (upsert, store) = Example::relearned(Lifecycle::Stale, check);

            assert_eq!(upsert, expected, "`{check}`");
            assert_eq!(
                Example::states(&store),
                [(1, Lifecycle::Active)],
                "`{check}`"
            );
            assert!(Example::recallable(&store), "`{check}`");
        }
    }

    #[test]
    fn indexes_files_by_path_name_and_stem() {
        let keys = |path: &str| Store::entity_keys(&Entity::Path(path.to_owned()));

        assert_eq!(
            keys("src/Index.js"),
            [
                ("src/index.js".to_owned(), "path"),
                ("index.js".to_owned(), "path"),
                ("index".to_owned(), "path"),
            ]
        );
        assert_eq!(keys("Makefile"), [("makefile".to_owned(), "path")]);
        assert_eq!(keys("db.rs"), [("db.rs".to_owned(), "path")]);
    }

    #[test]
    fn indexes_commands_by_the_words_a_prompt_would_use() {
        let keys = |command: &str| Store::entity_keys(&Entity::Command(command.to_owned()));

        assert_eq!(keys("npm test"), [("npm test".to_owned(), "command")]);
        assert_eq!(
            keys("Python -m pytest"),
            [("python m pytest".to_owned(), "command")]
        );
        assert!(keys("--").is_empty());
    }

    #[test]
    fn tells_a_file_name_stem_from_a_whole_file_name() {
        let mut procedure = Procedure::example();
        procedure.trigger.entities = vec![
            Entity::Path("src/Index.js".to_owned()),
            Entity::Path("Makefile".to_owned()),
            Entity::Path("makefile.am".to_owned()),
            Entity::Symbol("paginate".to_owned()),
        ];
        let hit = |key: &str, kind: &str| EntityHit {
            rowid: 1,
            key: key.to_owned(),
            kind: kind.to_owned(),
        };

        assert!(hit("index", "path").is_stem_of(&procedure));
        assert!(!hit("index.js", "path").is_stem_of(&procedure));
        assert!(!hit("src/index.js", "path").is_stem_of(&procedure));
        assert!(!hit("makefile", "path").is_stem_of(&procedure));
        assert!(!hit("paginate", "symbol").is_stem_of(&procedure));
    }

    #[test]
    fn lexical_search_is_not_crowded_out_by_other_repositories() {
        let mut zebras = Zebras::new();
        zebras.crowd(60).add(
            "here",
            Zebras::in_repo(REPO),
            "Render the zebra striped table rows",
        );

        assert_eq!(zebras.found(10), ["p_here"]);
    }

    #[test]
    fn lexical_search_ranks_only_recallable_procedures() {
        let mut zebras = Zebras::new();
        zebras
            .crowd(3)
            .add("global", Scope::Global, "Zebra stripes")
            .add(
                "here",
                Zebras::in_repo(REPO),
                "Render the zebra striped table rows",
            )
            .add("archived", Zebras::in_repo(REPO), "Zebra zebra zebra")
            .add("stale", Zebras::in_repo(REPO), "Zebra crossing")
            .add("candidate", Zebras::in_repo(REPO), "Zebra zebra");
        for (id, state) in [
            ("p_archived", Lifecycle::Archived),
            ("p_stale", Lifecycle::Stale),
            ("p_candidate", Lifecycle::Candidate),
        ] {
            zebras
                .0
                .set_states(id, &[(1, state)])
                .expect("state changes");
        }

        assert_eq!(zebras.found(10), ["p_global", "p_stale", "p_here"]);
        assert_eq!(zebras.found(2), ["p_global", "p_stale"]);
    }

    #[test]
    fn a_fix_and_an_add_in_the_same_files_become_separate_procedures() {
        let mut tasks = Tasks::new();

        assert_eq!(tasks.learn(Tasks::FIX), Upsert::Created { rowid: 1 });
        assert_eq!(tasks.learn(Tasks::ADD), Upsert::Created { rowid: 2 });

        let procedures = tasks.procedures();
        assert_ne!(procedures[0].0, procedures[1].0);
        assert_eq!(
            procedures
                .iter()
                .map(|(_, title)| title)
                .collect::<Vec<_>>(),
            [Tasks::FIX, Tasks::ADD]
        );
    }

    #[test]
    fn tasks_of_the_same_or_an_unclear_kind_share_a_procedure() {
        let mut tasks = Tasks::new();
        tasks.learn(Tasks::FIX);

        for prompt in [
            "The listing reports one page too few. Fix totalPages.",
            Tasks::UNCLEAR,
        ] {
            assert_eq!(
                tasks.learn(prompt),
                Upsert::Refreshed { rowid: 1 },
                "{prompt}"
            );
        }
        assert_eq!(tasks.procedures().len(), 1);
    }

    #[test]
    fn later_tasks_join_the_procedure_of_their_kind() {
        let mut tasks = Tasks::new();
        tasks.learn(Tasks::FIX);
        tasks.learn(Tasks::ADD);

        assert_eq!(
            tasks.learn("Support a descending order in paginate"),
            Upsert::Refreshed { rowid: 2 }
        );
        assert_eq!(
            tasks.learn("Page 3 is empty when the list has exactly 20 items. Fix it."),
            Upsert::Refreshed { rowid: 1 }
        );
        assert_eq!(tasks.procedures().len(), 2);
    }

    #[test]
    fn a_procedure_of_unclear_kind_takes_the_kind_of_the_first_task_that_joins_it() {
        let mut tasks = Tasks::new();

        assert_eq!(tasks.learn(Tasks::UNCLEAR), Upsert::Created { rowid: 1 });
        assert_eq!(tasks.learn(Tasks::ADD), Upsert::Refreshed { rowid: 1 });
        assert_eq!(tasks.learn(Tasks::FIX), Upsert::Created { rowid: 2 });
    }
}
