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
        let already = progress.and_then(|progress| progress.extracted_through);
        let last_seq = trace.events.last().map(|event| event.seq);
        if progress.is_some_and(|progress| progress.ended) && last_seq <= already {
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
            tasks.push(TaskSpan {
                first_seq: extraction.first_seq,
                last_seq: extraction.last_seq,
                outcome: extraction.outcome,
            });
            if already.is_some_and(|done| extraction.first_seq <= done) {
                continue;
            }
            report.tasks += 1;
            extracted_through = Some(extraction.last_seq);
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
                extracted_through: if ended {
                    last_seq.max(extracted_through)
                } else {
                    extracted_through
                },
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use serde_json::{Value, json};
    use trodden_store::{Cue, Injection};

    use super::*;

    const SESSION: &str = "0f6c1c4e-2f3a-4b8e-9d1a-5b2c3d4e5f60";
    const CWD: &str = "/home/dev/shop";

    struct Scratch {
        dir: PathBuf,
        transcript: PathBuf,
    }

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("trodden-{name}-{}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("scratch directory is writable");
            let transcript = dir.join("session.jsonl");
            Self { dir, transcript }
        }

        fn ingest(&self) -> Ingest {
            let home = Home::at(self.dir.join("home"));
            home.initialize().expect("home initializes");
            Ingest::start(&home).expect("ingest starts")
        }

        fn append(&self, lines: &[Value]) {
            let mut file = File::options()
                .create(true)
                .append(true)
                .open(&self.transcript)
                .expect("transcript is writable");
            for line in lines {
                writeln!(file, "{line}").expect("transcript line is written");
            }
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    fn at(minute: u32, second: u32) -> String {
        format!("2026-10-01T10:{minute:02}:{second:02}Z")
    }

    fn line(minute: u32, second: u32, kind: &str, message: Value) -> Value {
        json!({
            "type": kind,
            "sessionId": SESSION,
            "cwd": CWD,
            "timestamp": at(minute, second),
            "message": message,
        })
    }

    fn task(minute: u32, prompt: &str, file: &str) -> Vec<Value> {
        let path = format!("{CWD}/{file}");
        let edit = format!("edit-{minute}");
        let test = format!("test-{minute}");
        let mut edited = line(
            minute,
            20,
            "user",
            json!({"content": [{"type": "tool_result", "tool_use_id": edit, "content": "ok"}]}),
        );
        edited["toolUseResult"] = json!({
            "filePath": path,
            "originalFile": "module.exports = 1;\n",
            "structuredPatch": [{
                "oldStart": 1,
                "lines": ["-module.exports = 1;", "+module.exports = 2;"],
            }],
        });
        vec![
            line(minute, 0, "user", json!({"content": prompt})),
            line(
                minute,
                10,
                "assistant",
                json!({"content": [{"type": "tool_use", "id": edit, "name": "Edit", "input": {"file_path": path}}]}),
            ),
            edited,
            line(
                minute,
                30,
                "assistant",
                json!({"content": [{"type": "tool_use", "id": test, "name": "Bash", "input": {"command": "npm test"}}]}),
            ),
            line(
                minute,
                40,
                "user",
                json!({"content": [{"type": "tool_result", "tool_use_id": test, "content": "pass 3"}]}),
            ),
        ]
    }

    fn inject(ingest: &Ingest, minute: u32) {
        let row = ingest
            .store
            .list(None, true)
            .expect("procedures list")
            .remove(0);
        ingest
            .store
            .record_injection(&Injection {
                session: SESSION.to_owned(),
                procedure: row.procedure.id.to_string(),
                revision: row.procedure.revision,
                holdout: false,
                cue: Cue::Prompt,
                at: at(minute, 0).parse().expect("timestamp is valid"),
            })
            .expect("injection is recorded");
    }

    #[test]
    fn resumed_sessions_are_learned_from_after_ending() {
        let scratch = Scratch::new("resumed");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let first = ingest
            .claude_code(&scratch.transcript, true)
            .expect("first ingest");
        assert_eq!((first.sessions, first.tasks, first.created), (1, 1, 1));

        scratch.append(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        inject(&ingest, 5);
        let running = ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest while resumed");
        assert_eq!(
            (running.sessions, running.tasks, running.settled),
            (1, 0, 0)
        );

        let resumed = ingest
            .claude_code(&scratch.transcript, true)
            .expect("ingest after resuming");
        assert_eq!(
            (resumed.sessions, resumed.tasks, resumed.settled),
            (1, 1, 1)
        );

        let again = ingest
            .claude_code(&scratch.transcript, true)
            .expect("ingest with nothing new");
        assert_eq!(again, IngestReport::default());
    }

    #[test]
    fn follow_ups_after_resuming_join_the_extracted_task() {
        let scratch = Scratch::new("follow-up");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .claude_code(&scratch.transcript, true)
            .expect("first ingest");

        scratch.append(&task(5, "still failing", "src/paginate.js"));
        inject(&ingest, 5);
        let resumed = ingest
            .claude_code(&scratch.transcript, true)
            .expect("ingest after resuming");
        assert_eq!(
            (resumed.sessions, resumed.tasks, resumed.settled),
            (1, 0, 1)
        );

        let again = ingest
            .claude_code(&scratch.transcript, true)
            .expect("ingest with nothing new");
        assert_eq!(again, IngestReport::default());
    }
}
