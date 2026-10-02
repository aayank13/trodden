use anyhow::{Context, Result};
use jiff::Timestamp;
use rusqlite::{OptionalExtension, Transaction, TransactionBehavior, params};
use trodden_core::{
    Procedure,
    procedure::{Entity, Lifecycle, Scope, Slot},
};

use crate::{Store, Terms};

pub(crate) const LEXICAL_POSTINGS_BUDGET: i64 = 2_000;

const MAX_EXAMPLES: usize = 8;

const MAX_AVOID: usize = 3;

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

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LexicalHit {
    pub rowid: i64,
    pub score: f64,
}

impl Store {
    pub fn upsert(&mut self, candidate: &Procedure) -> Result<Upsert> {
        let signature = Self::signature(candidate)?;
        let tx = self
            .conn
            .transaction()
            .context("start storing a procedure")?;
        let existing: Vec<(i64, u32, String, String, String)> = {
            let mut statement = tx
                .prepare_cached("SELECT rowid, revision, signature, document, state FROM procedures WHERE family = ?1")
                .context("prepare the family lookup")?;
            statement
                .query_map([candidate.family.as_str()], |row| {
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
                .context("read the procedure family")?
        };

        let outcome = if let Some(mut absorber) = Self::absorber(&existing, candidate, &signature)?
        {
            let stored = &mut absorber.stored;
            Self::merge(stored, candidate);
            tx.execute(
                "UPDATE procedures SET document = ?1, signature = ?2, updated_at = ?3, used_at = ?4
                 WHERE rowid = ?5",
                params![
                    Self::document(stored)?,
                    Self::signature(stored)?,
                    stored.provenance.updated_at.to_string(),
                    Timestamp::now().to_string(),
                    absorber.rowid
                ],
            )
            .context("refresh a procedure revision")?;
            if matches!(absorber.state, Lifecycle::Active | Lifecycle::Stale) {
                Self::reindex(&tx, stored.id.as_str())?;
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
                Self::reindex(&tx, stored.id.as_str())?;
                Upsert::Created { rowid }
            } else {
                Upsert::Revised { rowid }
            }
        };
        tx.commit().context("commit a procedure")?;
        Ok(outcome)
    }

    fn absorber(
        family: &[(i64, u32, String, String, String)],
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
                Lifecycle::Active | Lifecycle::Stale | Lifecycle::Candidate
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
                let path = path.to_lowercase();
                let name = path.rsplit('/').next().unwrap_or(&path).to_owned();
                let stem = name.split('.').next().unwrap_or(&name).to_owned();
                let mut keys = vec![(path.clone(), "path")];
                if name != path {
                    keys.push((name.clone(), "path"));
                }
                if stem != name && stem.len() > 2 {
                    keys.push((stem, "path"));
                }
                keys
            }
            Entity::Symbol(symbol) => vec![(symbol.to_lowercase(), "symbol")],
            Entity::Command(command) => vec![(command.to_lowercase(), "command")],
            Entity::ErrorSignature(signature) => vec![(signature.to_lowercase(), "error")],
            Entity::Package(package) => vec![(package.to_lowercase(), "package")],
            _ => Vec::new(),
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
            .prepare("SELECT rowid, -rank FROM procedures_fts WHERE procedures_fts MATCH ?1 ORDER BY rank LIMIT ?2")
            .context("prepare the lexical search")?;
        let overfetch = i64::try_from(limit.saturating_mul(5)).unwrap_or(i64::MAX);
        let ranked: Vec<LexicalHit> = statement
            .query_map(params![query, overfetch], |row| {
                Ok(LexicalHit {
                    rowid: row.get(0)?,
                    score: row.get(1)?,
                })
            })
            .context("search procedure text")?
            .collect::<Result<_, _>>()
            .context("read lexical matches")?;

        let mut in_scope = self
            .conn
            .prepare("SELECT 1 FROM procedures WHERE rowid = ?1 AND state IN ('active', 'stale') AND (repo = ?2 OR repo = '')")
            .context("prepare the scope check")?;
        let mut hits = Vec::with_capacity(limit);
        for hit in ranked {
            if in_scope
                .exists(params![hit.rowid, repo])
                .context("check a match's scope")?
            {
                hits.push(hit);
                if hits.len() == limit {
                    break;
                }
            }
        }
        Ok(hits)
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
        let (condition, value) = match target {
            Forget::Procedure(id) => ("id = ?1", Some(id)),
            Forget::Repo(repo) => ("repo = ?1", Some(repo)),
            Forget::All => ("?1 IS NULL", None),
        };
        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .context("start forgetting")?;
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
            tx.execute_batch(
                "DELETE FROM extractions; DELETE FROM sessions; DELETE FROM injections;",
            )
            .context("delete ingest history")?;
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
