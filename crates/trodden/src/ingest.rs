use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use jiff::Timestamp;
use trodden_capture::claude_code::{HARNESS, Transcript};
use trodden_embed::{Embedder, Quantized};
use trodden_extract::{Extractor, ProjectChecks};
use trodden_recall::{Skeleton, VectorIndex};
use trodden_redact::Redactor;
use trodden_store::{ExtractionRecord, Patience, Progress, Store, Upsert};

use crate::{
    Home, Workspace,
    learning::{Changes, Learning, TaskSpan},
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IngestReport {
    pub sessions: usize,
    pub tasks: usize,
    pub created: usize,
    pub refreshed: usize,
    pub revised: usize,
    pub rejected: BTreeMap<String, usize>,
    pub settled: usize,
    pub promoted: usize,
    pub quarantined: usize,
    pub aged: usize,
}

impl IngestReport {
    fn count(&mut self, upsert: Upsert) {
        match upsert {
            Upsert::Created { .. } => self.created += 1,
            Upsert::Refreshed { .. } | Upsert::Generalized { .. } => self.refreshed += 1,
            Upsert::Revised { .. } => self.revised += 1,
        }
    }

    fn absorb(&mut self, other: Self) {
        self.sessions += other.sessions;
        self.tasks += other.tasks;
        self.created += other.created;
        self.refreshed += other.refreshed;
        self.revised += other.revised;
        for (reason, count) in other.rejected {
            *self.rejected.entry(reason).or_default() += count;
        }
        self.settled += other.settled;
        self.promoted += other.promoted;
        self.quarantined += other.quarantined;
        self.aged += other.aged;
    }

    fn absorb_changes(&mut self, changes: &Changes) {
        self.promoted += changes.promoted;
        self.quarantined += changes.quarantined;
        self.aged += changes.aged;
    }
}

#[derive(Debug)]
pub struct Ingest {
    home: Home,
    store: Store,
    embedder: Option<Embedder>,
    redactor: Redactor,
    _lock: File,
}

impl Ingest {
    const IDLE_SECONDS: u64 = 30 * 60;

    pub fn start(home: &Home) -> Result<Self> {
        let lock = File::create(home.ingest_lock()).context("create the ingest lock")?;
        lock.lock().context("wait for the ingest lock")?;
        let mut ingest = Self {
            home: home.clone(),
            store: home.open_store(Patience::Batch)?,
            embedder: home.embedder(),
            redactor: Redactor::new(),
            _lock: lock,
        };
        if ingest.embedder.is_some()
            && ingest.store.embedding_scheme()?.as_deref() != Some(Self::EMBEDDING_SCHEME)
        {
            ingest.reembed()?;
        }
        Ok(ingest)
    }

    const EMBEDDING_SCHEME: &str = "prompt-skeletons";

    pub fn claude_code(&mut self, transcript: &Path, ended: bool) -> Result<IngestReport> {
        if self.store.paused()? {
            return Ok(IngestReport::default());
        }
        let text = fs::read_to_string(transcript)
            .with_context(|| format!("read {}", transcript.display()))?;
        let mut trace = Transcript::parse(&text, &self.redactor)
            .with_context(|| format!("parse {}", transcript.display()))?;
        let ended = ended || Self::is_idle(transcript);
        let session = trace.session.as_str().to_owned();
        let progress = self.store.progress(&session)?;
        if progress.is_some_and(|progress| progress.ended) {
            return Ok(IngestReport::default());
        }

        let cwd = Self::expand_home(&trace.cwd);
        ensure!(
            cwd.is_absolute() && !trace.cwd.contains("[REDACTED"),
            "{} has no usable working directory",
            transcript.display()
        );
        let workspace = Workspace::resolve(&cwd, &self.store)?;
        trace.commit = workspace.head();
        let already = progress.and_then(|progress| progress.extracted_through);
        let mut report = IngestReport {
            sessions: 1,
            ..IngestReport::default()
        };
        let mut extracted_through = already;
        let now = Timestamp::now();
        let stamp = now.to_string();
        let mut tasks = Vec::new();

        let checks = ProjectChecks::discover(&workspace.root)
            .redacted(|command| self.redactor.redact(command).into_owned());
        for extraction in Extractor::new(checks).extract(&trace, &workspace.repo, ended) {
            if already.is_some_and(|done| extraction.last_seq <= done) {
                continue;
            }
            report.tasks += 1;
            extracted_through = Some(extraction.last_seq);
            tasks.push(TaskSpan {
                first_seq: extraction.first_seq,
                last_seq: extraction.last_seq,
                outcome: extraction.outcome,
            });
            let (procedure, rejection) = match extraction.result {
                Ok(procedure) => {
                    let upsert = self.store.upsert(&procedure)?;
                    report.count(upsert);
                    self.embed_row(upsert.rowid())?;
                    (Some(upsert.rowid()), None)
                }
                Err(rejection) => {
                    let reason = rejection.to_string();
                    *report
                        .rejected
                        .entry(Self::reason_kind(&reason))
                        .or_default() += 1;
                    (None, Some(reason))
                }
            };
            self.store.record_extraction(&ExtractionRecord {
                session: session.clone(),
                first_seq: extraction.first_seq,
                summary: extraction.summary,
                procedure,
                rejection,
                outcome: Some(extraction.outcome.as_str().to_owned()),
                tool_calls: Some(extraction.tool_calls),
                span: Some((
                    extraction.started_at.to_string(),
                    extraction.ended_at.to_string(),
                )),
                at: stamp.clone(),
            })?;
        }

        let mut learning = Learning::new(&mut self.store);
        let (settled, touched) = learning.settle_injections(&trace, &tasks)?;
        report.settled = settled;
        let settled = learning.settle(&touched)?;
        let aged = learning.age(now)?;
        report.absorb_changes(&settled);
        report.absorb_changes(&aged);

        self.store.set_progress(
            &session,
            HARNESS,
            &transcript.to_string_lossy(),
            Some(workspace.repo.as_str()),
            Progress {
                extracted_through,
                ended,
            },
            &stamp,
        )?;
        if report.created + report.refreshed + report.revised > 0 || settled.any() || aged.any() {
            self.rebuild_index()?;
        }
        Ok(report)
    }

    pub fn backfill_claude_code(
        &mut self,
        projects: &Path,
    ) -> Result<(IngestReport, Vec<(PathBuf, String)>)> {
        let mut transcripts = Vec::new();
        for project in
            fs::read_dir(projects).with_context(|| format!("list {}", projects.display()))?
        {
            let project = project.context("read a project directory entry")?.path();
            let Ok(entries) = fs::read_dir(&project) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .extension()
                    .is_some_and(|extension| extension == "jsonl")
                {
                    transcripts.push(path);
                }
            }
        }
        transcripts.sort();

        let mut report = IngestReport::default();
        let mut failures = Vec::new();
        for transcript in transcripts {
            match self.claude_code(&transcript, false) {
                Ok(one) => report.absorb(one),
                Err(error) => failures.push((transcript, format!("{error:#}"))),
            }
        }
        Ok((report, failures))
    }

    pub fn reembed(&mut self) -> Result<usize> {
        self.embedder = Some(Embedder::open(&self.home.embeddings())?);
        let rows = self.store.list(None, true)?;
        for row in &rows {
            self.embed_row(row.rowid)?;
        }
        self.store.set_embedding_scheme(Self::EMBEDDING_SCHEME)?;
        self.rebuild_index()?;
        Ok(rows.len())
    }

    pub fn rebuild_index(&self) -> Result<()> {
        VectorIndex::build(&self.store.recallable_embeddings()?, &self.home.index())
    }

    fn embed_row(&mut self, rowid: i64) -> Result<()> {
        let Some(embedder) = &mut self.embedder else {
            return Ok(());
        };
        let Some(row) = self.store.procedure(rowid)? else {
            return Ok(());
        };
        let trigger = &row.procedure.trigger;
        let mut embeddings = Vec::new();
        for prompt in std::iter::once(&trigger.text).chain(&trigger.examples) {
            let skeleton = Skeleton::of(prompt);
            if let Some(embedding) = embedder
                .embed(skeleton.as_str())
                .context("embed a procedure")?
            {
                embeddings.push(Quantized::new(&embedding).to_bytes());
            }
        }
        self.store.set_embeddings(rowid, &embeddings)
    }

    fn is_idle(transcript: &Path) -> bool {
        fs::metadata(transcript)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| modified.elapsed().ok())
            .is_some_and(|age| age.as_secs() > Self::IDLE_SECONDS)
    }

    fn expand_home(path: &str) -> PathBuf {
        match (path.strip_prefix('~'), std::env::var_os("HOME")) {
            (Some(rest), Some(home)) => PathBuf::from(format!("{}{rest}", home.to_string_lossy())),
            _ => PathBuf::from(path),
        }
    }

    fn reason_kind(reason: &str) -> String {
        reason.split(':').next().unwrap_or(reason).to_owned()
    }
}
