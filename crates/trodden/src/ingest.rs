use std::{
    collections::BTreeMap,
    fs::{self, File, TryLockError},
    io::Write,
    path::{self, Path, PathBuf},
    process,
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use jiff::Timestamp;
use serde::{Deserialize, Serialize};
use trodden_core::Skeleton;
use trodden_embed::{Embedder, Quantized};
use trodden_extract::{Extractor, ProjectChecks};
use trodden_recall::VectorIndex;
use trodden_redact::Redactor;
use trodden_store::{ExtractionRecord, Patience, Progress, Store, Upsert};

use crate::{
    Harness, Home, Workspace,
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

    pub fn absorb(&mut self, other: Self) {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct PendingIngest {
    harness: String,
    transcript: PathBuf,
    ended: bool,
}

#[derive(Debug)]
struct PendingIngests {
    home: Home,
    dir: PathBuf,
}

impl PendingIngests {
    const EXTENSION: &str = "marker";

    fn new(home: &Home) -> Self {
        Self {
            home: home.clone(),
            dir: home.pending_ingests(),
        }
    }

    fn push(&self, harness: Harness, transcript: &Path, ended: bool) -> Result<()> {
        let pending = PendingIngest {
            harness: harness.as_str().to_owned(),
            transcript: path::absolute(transcript)
                .with_context(|| format!("resolve {}", transcript.display()))?,
            ended,
        };
        fs::create_dir_all(&self.dir).with_context(|| format!("create {}", self.dir.display()))?;
        static QUEUED: AtomicUsize = AtomicUsize::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let name = format!(
            "{nanos}-{}-{}",
            process::id(),
            QUEUED.fetch_add(1, Ordering::Relaxed)
        );
        let staged = self.dir.join(format!("{name}.tmp"));
        let bytes = serde_json::to_vec(&pending).context("encode a pending ingest")?;
        File::create_new(&staged)
            .and_then(|mut file| file.write_all(&bytes))
            .with_context(|| format!("write {}", staged.display()))?;
        let marker = self.dir.join(format!("{name}.{}", Self::EXTENSION));
        fs::rename(&staged, &marker).with_context(|| format!("queue {}", marker.display()))
    }

    fn any(&self) -> Result<bool> {
        Ok(!self.markers()?.is_empty())
    }

    fn take(&self) -> Result<Vec<PendingIngest>> {
        let mut transcripts = BTreeMap::new();
        for marker in self.markers()? {
            let text = fs::read(&marker).with_context(|| format!("read {}", marker.display()))?;
            let pending = serde_json::from_slice::<PendingIngest>(&text)
                .with_context(|| format!("parse the queued ingest {}", marker.display()));
            if let Err(error) = &pending {
                self.home.log_error(error);
            }
            fs::remove_file(&marker).with_context(|| format!("remove {}", marker.display()))?;
            if let Ok(pending) = pending {
                *transcripts
                    .entry((pending.harness, pending.transcript))
                    .or_default() |= pending.ended;
            }
        }
        Ok(transcripts
            .into_iter()
            .map(|((harness, transcript), ended)| PendingIngest {
                harness,
                transcript,
                ended,
            })
            .collect())
    }

    fn markers(&self) -> Result<Vec<PathBuf>> {
        let entries = match fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(error).with_context(|| format!("list {}", self.dir.display()));
            }
        };
        let mut markers = Vec::new();
        for entry in entries {
            let path = entry
                .with_context(|| format!("read an entry of {}", self.dir.display()))?
                .path();
            if path
                .extension()
                .is_some_and(|extension| extension == Self::EXTENSION)
            {
                markers.push(path);
            }
        }
        markers.sort();
        Ok(markers)
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
    const IDLE_SESSIONS_PER_RUN: usize = 4;

    pub fn start(home: &Home) -> Result<Self> {
        let lock = File::create(home.ingest_lock()).context("create the ingest lock")?;
        lock.lock().context("wait for the ingest lock")?;
        Self::holding(home, lock)
    }

    fn try_start(home: &Home) -> Result<Option<Self>> {
        let lock = File::create(home.ingest_lock()).context("create the ingest lock")?;
        match lock.try_lock() {
            Ok(()) => Self::holding(home, lock).map(Some),
            Err(TryLockError::WouldBlock) => Ok(None),
            Err(TryLockError::Error(error)) => Err(error).context("check the ingest lock"),
        }
    }

    fn holding(home: &Home, lock: File) -> Result<Self> {
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

    pub fn run(
        home: &Home,
        harness: Harness,
        transcript: &Path,
        ended: bool,
    ) -> Result<IngestReport> {
        let mut ingest = Self::start(home)?;
        let report = ingest.transcript(harness, transcript, ended);
        let drained = ingest.finish();
        let report = report?;
        drained?;
        Ok(report)
    }

    pub fn run_or_defer(
        home: &Home,
        harness: Harness,
        transcript: &Path,
        ended: bool,
    ) -> Result<Option<IngestReport>> {
        PendingIngests::new(home).push(harness, transcript, ended)?;
        Self::try_start(home)?.map(Self::finish).transpose()
    }

    fn finish(mut self) -> Result<IngestReport> {
        let mut report = self.drain()?;
        let home = self.home.clone();
        drop(self);
        report.absorb(Self::pick_up(&home)?);
        Ok(report)
    }

    fn pick_up(home: &Home) -> Result<IngestReport> {
        let mut report = IngestReport::default();
        while PendingIngests::new(home).any()? {
            let Some(mut ingest) = Self::try_start(home)? else {
                break;
            };
            report.absorb(ingest.drain()?);
        }
        Ok(report)
    }

    fn drain(&mut self) -> Result<IngestReport> {
        let pending = PendingIngests::new(&self.home);
        let mut report = IngestReport::default();
        loop {
            let batch = pending.take()?;
            if batch.is_empty() {
                return Ok(report);
            }
            for PendingIngest {
                harness,
                transcript,
                ended,
            } in batch
            {
                let Some(known) = Harness::from_name(&harness) else {
                    self.home.log_error(format_args!(
                        "ingest {}: unsupported harness `{harness}`",
                        transcript.display()
                    ));
                    continue;
                };
                match self.transcript(known, &transcript, ended) {
                    Ok(one) => report.absorb(one),
                    Err(error) => self.home.log_error(error),
                }
            }
        }
    }

    fn transcript(
        &mut self,
        harness: Harness,
        transcript: &Path,
        ended: bool,
    ) -> Result<IngestReport> {
        let mut report = self
            .session(harness, transcript, ended)
            .with_context(|| format!("ingest {}", transcript.display()))?;
        report.absorb(self.finish_idle_sessions(harness)?);
        Ok(report)
    }

    fn finish_idle_sessions(&mut self, harness: Harness) -> Result<IngestReport> {
        let mut report = IngestReport::default();
        if self.store.paused()? {
            return Ok(report);
        }
        let idle: Vec<_> = self
            .store
            .open_sessions(harness.as_str())?
            .into_iter()
            .map(|(session, transcript)| (session, PathBuf::from(transcript)))
            .filter(|(_, transcript)| Self::is_idle(transcript) || !transcript.exists())
            .take(Self::IDLE_SESSIONS_PER_RUN)
            .collect();
        for (session, transcript) in idle {
            match self.session(harness, &transcript, true) {
                Ok(finished) => report.absorb(finished),
                Err(_) => self
                    .store
                    .end_session(&session, &Timestamp::now().to_string())?,
            }
        }
        Ok(report)
    }

    fn session(
        &mut self,
        harness: Harness,
        transcript: &Path,
        ended: bool,
    ) -> Result<IngestReport> {
        let text = Harness::read(transcript)?;
        let (mut trace, cwd) = harness
            .parse(&text, &self.redactor)
            .context("parse the transcript")?;
        let ended = ended || Self::is_idle(transcript);
        let session = trace.session.as_str().to_owned();
        let progress = self.store.progress(&session)?;
        let already = progress.and_then(|progress| progress.extracted_through);
        let last_seq = trace.events.last().map(|event| event.seq);
        if progress.is_some_and(|progress| progress.ended) && last_seq <= already {
            return Ok(IngestReport::default());
        }

        let cwd = cwd
            .filter(|cwd| cwd.is_absolute())
            .context("the transcript has no usable working directory")?;
        let workspace = Workspace::resolve(&cwd, &self.store)?;
        let recorded = path::absolute(transcript)
            .with_context(|| format!("resolve {}", transcript.display()))?;
        let recorded = recorded.to_string_lossy();
        let now = Timestamp::now();
        let stamp = now.to_string();
        if self.store.paused()? {
            self.store.set_progress(
                &session,
                harness.as_str(),
                &recorded,
                Some(workspace.repo.as_str()),
                Progress {
                    extracted_through: last_seq.max(already),
                    ended,
                },
                &stamp,
            )?;
            if ended {
                self.drop_journal(harness, &session, transcript)?;
            }
            return Ok(IngestReport::default());
        }
        if trace.commit.is_none() {
            let last_activity = trace
                .events
                .last()
                .map_or(trace.started_at, |event| event.at);
            trace.commit = workspace.head_at(last_activity);
        }
        let mut report = IngestReport {
            sessions: 1,
            ..IngestReport::default()
        };
        let mut extracted_through = already;
        let mut tasks = Vec::new();
        let forgotten = self.store.forgotten_before()?;
        let done = |seq: u32, at: Timestamp| {
            already.is_some_and(|through| seq <= through)
                || forgotten.is_some_and(|forgot| at <= forgot)
        };

        let checks = ProjectChecks::discover(&workspace.root)
            .redacted(|command| self.redactor.redact(command).into_owned());
        for extraction in Extractor::new(checks).extract(&trace, &workspace.repo, ended) {
            if done(extraction.last_seq, extraction.ended_at) {
                continue;
            }
            tasks.push(TaskSpan {
                first_seq: extraction.first_seq,
                last_seq: extraction.last_seq,
                outcome: extraction.outcome,
            });
            if done(extraction.first_seq, extraction.started_at) {
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
            harness.as_str(),
            &recorded,
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
        if ended {
            self.drop_journal(harness, &session, transcript)?;
        }
        Ok(report)
    }

    // A resumed conversation starts a new journal numbered from zero, so its progress goes too.
    fn drop_journal(&self, harness: Harness, session: &str, journal: &Path) -> Result<()> {
        if !harness.journaled() {
            return Ok(());
        }
        match fs::remove_file(journal) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => {
                return Err(error).with_context(|| format!("remove {}", journal.display()));
            }
            _ => {}
        }
        self.store.forget_session(session)
    }

    pub fn backfill(
        &mut self,
        harness: Harness,
        history: &Path,
    ) -> Result<(IngestReport, Vec<(PathBuf, String)>)> {
        if self.store.paused()? {
            return Ok((IngestReport::default(), Vec::new()));
        }
        let mut report = IngestReport::default();
        let mut failures = Vec::new();
        for transcript in harness.transcripts(history)? {
            match self.session(harness, &transcript, false) {
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

    fn reason_kind(reason: &str) -> String {
        reason.split(':').next().unwrap_or(reason).to_owned()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};
    use trodden_store::{Cue, Forget, Injection};

    use super::*;

    const SESSION: &str = "0f6c1c4e-2f3a-4b8e-9d1a-5b2c3d4e5f60";
    const OTHER_SESSION: &str = "1a7d2d5f-3a4b-4c9f-8e2b-6c3d4e5f6a71";
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

        fn home(&self) -> Home {
            let home = Home::at(self.dir.join("home"));
            home.initialize().expect("home initializes");
            home
        }

        fn ingest(&self) -> Ingest {
            Ingest::start(&self.home()).expect("ingest starts")
        }

        fn numbered_session(&self, index: usize) -> (String, PathBuf) {
            let session = format!("0f6c1c4e-2f3a-4b8e-9d1a-{index:012}");
            let transcript = self.dir.join(format!("{session}.jsonl"));
            let text: String = Self::task(
                0,
                &format!("The total in src/order{index}.js ignores the discount code"),
                &format!("src/order{index}.js"),
            )
            .iter()
            .map(|line| format!("{}\n", line.to_string().replace(SESSION, &session)))
            .collect();
            fs::write(&transcript, text).expect("transcript is writable");
            (session, transcript)
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

        fn append_damaged(&self, lines: &[Value]) {
            let mut file = File::options()
                .create(true)
                .append(true)
                .open(&self.transcript)
                .expect("transcript is writable");
            for line in lines {
                let latin1 = line
                    .to_string()
                    .split('\u{e9}')
                    .map(str::as_bytes)
                    .collect::<Vec<_>>()
                    .join(&0xe9);
                file.write_all(&latin1)
                    .and_then(|()| file.write_all(b"\n"))
                    .expect("transcript line is written");
            }
        }

        fn tear(&self) {
            File::options()
                .append(true)
                .open(&self.transcript)
                .and_then(|mut file| {
                    file.write_all(b"{\"type\":\"assistant\",\"message\":{\"content\":\"caf\xc3")
                })
                .expect("torn line is written");
        }

        fn append_in(&self, cwd: &str, lines: &[Value]) {
            let moved: Vec<Value> = lines
                .iter()
                .map(|line| {
                    serde_json::from_str(&line.to_string().replace(CWD, cwd))
                        .expect("moved line is JSON")
                })
                .collect();
            self.append(&moved);
        }

        fn repository(&self, committed: &[&str]) -> (PathBuf, Vec<String>) {
            let repo = self.dir.join("shop");
            fs::create_dir_all(&repo).expect("repository directory is writable");
            Self::git(&repo, &[], &["init", "-q"]);
            let commits = committed
                .iter()
                .map(|time| {
                    fs::write(repo.join("CHANGES"), time).expect("file is writable");
                    Self::git(&repo, &[], &["add", "-A"]);
                    let dated = [("GIT_COMMITTER_DATE", *time), ("GIT_AUTHOR_DATE", *time)];
                    Self::git(&repo, &dated, &["commit", "-q", "-m", time]);
                    Self::git(&repo, &[], &["rev-parse", "HEAD"])
                })
                .collect();
            (repo, commits)
        }

        fn git(repo: &Path, env: &[(&str, &str)], args: &[&str]) -> String {
            let output = process::Command::new("git")
                .arg("-C")
                .arg(repo)
                .args([
                    "-c",
                    "user.name=Dev",
                    "-c",
                    "user.email=dev@example.com",
                    "-c",
                    "commit.gpgsign=false",
                ])
                .args(args)
                .envs(env.iter().copied())
                .output()
                .expect("git runs");
            assert!(
                output.status.success(),
                "git {args:?} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8_lossy(&output.stdout).trim().to_owned()
        }

        fn commits(ingest: &Ingest) -> Vec<Option<String>> {
            ingest
                .store
                .list(None, true)
                .expect("procedures list")
                .into_iter()
                .flat_map(|row| row.procedure.provenance.sources)
                .map(|source| source.commit)
                .collect()
        }

        fn other_session(&self, lines: &[Value]) -> PathBuf {
            let transcript = self.dir.join("other.jsonl");
            let text: String = lines
                .iter()
                .map(|line| format!("{}\n", line.to_string().replace(SESSION, OTHER_SESSION)))
                .collect();
            fs::write(&transcript, text).expect("transcript is writable");
            transcript
        }

        fn go_idle(&self) {
            let long_ago = std::time::SystemTime::now()
                - std::time::Duration::from_secs(Ingest::IDLE_SECONDS + 60);
            File::options()
                .write(true)
                .open(&self.transcript)
                .and_then(|file| file.set_modified(long_ago))
                .expect("transcript mtime is set");
        }

        fn ended(ingest: &Ingest) -> bool {
            ingest
                .store
                .progress(SESSION)
                .expect("progress reads")
                .expect("session is recorded")
                .ended
        }

        fn logged(home: &Home) -> Vec<String> {
            fs::read_to_string(home.error_log())
                .unwrap_or_default()
                .lines()
                .map(|line| {
                    let (_, message) = line.split_once(' ').expect("line has a timestamp");
                    message.to_owned()
                })
                .collect()
        }

        fn stored(&self) -> String {
            ["trodden.db", "trodden.db-wal"]
                .iter()
                .map(|name| fs::read(self.dir.join("home").join(name)).unwrap_or_default())
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .collect()
        }

        fn at(minute: u32, second: u32) -> String {
            format!("2026-10-01T10:{minute:02}:{second:02}Z")
        }

        fn line(minute: u32, second: u32, kind: &str, message: Value) -> Value {
            json!({
                "type": kind,
                "sessionId": SESSION,
                "cwd": CWD,
                "timestamp": Self::at(minute, second),
                "message": message,
            })
        }

        fn task(minute: u32, prompt: &str, file: &str) -> Vec<Value> {
            let path = format!("{CWD}/{file}");
            let edit = format!("edit-{minute}");
            let test = format!("test-{minute}");
            let mut edited = Self::line(
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
                Self::line(minute, 0, "user", json!({"content": prompt})),
                Self::line(
                    minute,
                    10,
                    "assistant",
                    json!({"content": [{"type": "tool_use", "id": edit, "name": "Edit", "input": {"file_path": path}}]}),
                ),
                edited,
                Self::line(
                    minute,
                    30,
                    "assistant",
                    json!({"content": [{"type": "tool_use", "id": test, "name": "Bash", "input": {"command": "npm test"}}]}),
                ),
                Self::line(
                    minute,
                    40,
                    "user",
                    json!({"content": [{"type": "tool_result", "tool_use_id": test, "content": "pass 3"}]}),
                ),
            ]
        }

        fn tomorrow(lines: &[Value]) -> Vec<Value> {
            let tomorrow = Timestamp::now()
                .checked_add(jiff::SignedDuration::from_hours(24))
                .expect("tomorrow is representable")
                .strftime("%Y-%m-%d")
                .to_string();
            lines
                .iter()
                .map(|line| {
                    serde_json::from_str(&line.to_string().replace("2026-10-01", &tomorrow))
                        .expect("moved line is JSON")
                })
                .collect()
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
                    at: Self::at(minute, 0).parse().expect("timestamp is valid"),
                })
                .expect("injection is recorded");
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.dir);
        }
    }

    #[test]
    fn resumed_sessions_are_learned_from_after_ending() {
        let scratch = Scratch::new("resumed");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let first = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("first ingest");
        assert_eq!((first.sessions, first.tasks, first.created), (1, 1, 1));

        scratch.append(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        Scratch::inject(&ingest, 5);
        let running = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest while resumed");
        assert_eq!(
            (running.sessions, running.tasks, running.settled),
            (1, 0, 0)
        );

        let resumed = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest after resuming");
        assert_eq!(
            (resumed.sessions, resumed.tasks, resumed.settled),
            (1, 1, 1)
        );

        let again = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest with nothing new");
        assert_eq!(again, IngestReport::default());
    }

    #[test]
    fn follow_ups_after_resuming_join_the_extracted_task() {
        let scratch = Scratch::new("follow-up");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("first ingest");

        scratch.append(&Scratch::task(5, "still failing", "src/paginate.js"));
        Scratch::inject(&ingest, 5);
        let resumed = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest after resuming");
        assert_eq!(
            (resumed.sessions, resumed.tasks, resumed.settled),
            (1, 0, 1)
        );

        let again = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest with nothing new");
        assert_eq!(again, IngestReport::default());
    }

    #[test]
    fn work_done_while_paused_is_never_learned() {
        let scratch = Scratch::new("paused");
        let mut ingest = scratch.ingest();
        ingest.store.set_paused(true).expect("capture pauses");
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let paused = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest while paused");
        assert_eq!(paused, IngestReport::default());

        ingest.store.set_paused(false).expect("capture resumes");
        let ended = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest after resuming capture");
        assert_eq!((ended.sessions, ended.tasks, ended.created), (1, 0, 0));

        scratch.append(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let resumed = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest a task done after resuming capture");
        assert_eq!(
            (resumed.sessions, resumed.tasks, resumed.created),
            (1, 1, 1)
        );
        let titles: Vec<_> = ingest
            .store
            .list(None, true)
            .expect("procedures list")
            .into_iter()
            .map(|row| row.procedure.title)
            .collect();
        assert_eq!(
            titles,
            ["The cart total in src/cart.js ignores the discount code"]
        );
    }

    #[test]
    fn live_sessions_do_not_relearn_what_was_forgotten() {
        let scratch = Scratch::new("forget-live");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        scratch.append(&Scratch::task(
            5,
            "Checkout in src/checkout.js charges shipping twice",
            "src/checkout.js",
        ));
        let learned = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest before forgetting");
        assert_eq!((learned.tasks, learned.created), (1, 1));

        ingest
            .store
            .forget(Forget::All)
            .expect("everything is forgotten");
        scratch.append(&Scratch::tomorrow(&Scratch::task(
            10,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        )));
        scratch.append(&Scratch::tomorrow(&Scratch::task(
            15,
            "The search in src/search.js drops accented names",
            "src/search.js",
        )));
        let next = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest after forgetting");

        assert_eq!((next.tasks, next.created), (1, 1));
        let titles: Vec<_> = ingest
            .store
            .list(None, true)
            .expect("procedures list")
            .into_iter()
            .map(|row| row.procedure.title)
            .collect();
        assert_eq!(
            titles,
            ["The cart total in src/cart.js ignores the discount code"]
        );
    }

    #[test]
    fn sessions_spanning_a_pause_learn_only_what_followed_it() {
        let scratch = Scratch::new("spanning-pause");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("first ingest");

        scratch.append(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        Scratch::inject(&ingest, 5);
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest before pausing");

        ingest.store.set_paused(true).expect("capture pauses");
        scratch.append(&Scratch::task(
            10,
            "Search in src/search.js returns archived products to shoppers",
            "src/search.js",
        ));
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest while paused");

        ingest.store.set_paused(false).expect("capture resumes");
        scratch.append(&Scratch::task(15, "still failing", "src/search.js"));
        Scratch::inject(&ingest, 15);
        scratch.append(&Scratch::task(
            20,
            "Checkout in src/checkout.js charges shipping twice",
            "src/checkout.js",
        ));
        let resumed = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest after resuming capture");
        assert_eq!((resumed.tasks, resumed.created, resumed.settled), (1, 1, 1));

        let settled: Vec<_> = ingest
            .store
            .injections(Some(SESSION))
            .expect("injections list")
            .into_iter()
            .map(|record| record.outcome.is_some())
            .collect();
        assert_eq!(settled, [false, true]);
    }

    #[test]
    fn sessions_in_directories_that_look_like_secrets_are_learned() {
        let email = "jane.doe@gmail.com";
        let drive = format!("/Users/jane/Library/CloudStorage/GoogleDrive-{email}/shop");
        let worktree = "/home/dev/store/.claude/worktrees/agent-a3f9c2d17e5b4c08";
        for (name, cwd) in [("drive", drive.as_str()), ("worktree", worktree)] {
            let scratch = Scratch::new(name);
            let mut ingest = scratch.ingest();
            scratch.append_in(
                cwd,
                &Scratch::task(
                    0,
                    &format!("The cart total in {cwd}/src/cart.js ignores the discount code"),
                    "src/cart.js",
                ),
            );
            let report = ingest
                .transcript(Harness::ClaudeCode, &scratch.transcript, true)
                .expect("ingest in an unusual directory");
            assert_eq!((report.sessions, report.tasks, report.created), (1, 1, 1));
            assert!(!scratch.stored().contains(email), "{cwd}");
        }
    }

    #[test]
    fn sessions_without_an_absolute_directory_are_refused() {
        let scratch = Scratch::new("relative-cwd");
        let mut ingest = scratch.ingest();
        scratch.append_in(
            "shop",
            &Scratch::task(0, "Fix the cart total", "src/cart.js"),
        );
        let error = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect_err("a relative directory is refused");
        assert_eq!(
            format!("{error:#}"),
            format!(
                "ingest {}: the transcript has no usable working directory",
                scratch.transcript.display()
            )
        );
    }

    #[test]
    fn transcripts_with_invalid_utf8_are_still_learned() {
        let scratch = Scratch::new("invalid-utf8");
        let mut ingest = scratch.ingest();
        scratch.append_damaged(&Scratch::task(
            0,
            "The cart total in src/caf\u{e9}.js ignores the discount code",
            "src/cart.js",
        ));
        scratch.tear();
        let report = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("ingest a damaged transcript");
        assert_eq!((report.sessions, report.tasks, report.created), (1, 1, 1));
        let titles: Vec<_> = ingest
            .store
            .list(None, true)
            .expect("procedures list")
            .into_iter()
            .map(|row| row.procedure.title)
            .collect();
        assert_eq!(
            titles,
            ["The cart total in src/caf\u{fffd}.js ignores the discount code"]
        );
    }

    #[test]
    fn idle_sessions_with_invalid_utf8_are_learned_before_ending() {
        let scratch = Scratch::new("idle-invalid-utf8");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest before the transcript is damaged");
        scratch.tear();
        scratch.go_idle();

        let other = scratch.other_session(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let swept = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest after the damaged session went idle");
        assert_eq!((swept.sessions, swept.tasks, swept.created), (2, 1, 1));
        assert!(Scratch::ended(&ingest));
    }

    #[test]
    fn backfill_learns_transcripts_with_invalid_utf8() {
        let scratch = Scratch::new("backfill-invalid-utf8");
        let mut ingest = scratch.ingest();
        scratch.append_damaged(&Scratch::task(
            0,
            "The cart total in src/caf\u{e9}.js ignores the discount code",
            "src/cart.js",
        ));
        scratch.append(&Scratch::task(
            5,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        scratch.tear();
        let projects = scratch.dir.join("projects");
        fs::create_dir_all(projects.join("shop")).expect("projects directory is writable");
        fs::copy(&scratch.transcript, projects.join("shop/session.jsonl"))
            .expect("transcript is copied");
        let (report, failures) = ingest
            .backfill(Harness::ClaudeCode, &projects)
            .expect("backfill damaged history");
        assert!(failures.is_empty(), "{failures:?}");
        assert_eq!((report.sessions, report.created), (1, 1));
    }

    #[test]
    fn backfill_while_paused_leaves_history_for_later() {
        let scratch = Scratch::new("paused-backfill");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        scratch.append(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let projects = scratch.dir.join("projects");
        fs::create_dir_all(projects.join("shop")).expect("projects directory is writable");
        fs::copy(&scratch.transcript, projects.join("shop/session.jsonl"))
            .expect("transcript is copied");

        ingest.store.set_paused(true).expect("capture pauses");
        let (paused, _) = ingest
            .backfill(Harness::ClaudeCode, &projects)
            .expect("backfill while paused");
        assert_eq!(paused, IngestReport::default());
        assert_eq!(
            ingest.store.progress(SESSION).expect("progress reads"),
            None
        );

        ingest.store.set_paused(false).expect("capture resumes");
        let (resumed, failures) = ingest
            .backfill(Harness::ClaudeCode, &projects)
            .expect("backfill after resuming capture");
        assert!(failures.is_empty());
        assert_eq!((resumed.sessions, resumed.created), (1, 1));
    }

    #[test]
    fn idle_sessions_that_never_ended_are_finished_by_later_ingests() {
        let scratch = Scratch::new("never-ended");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let crashed = ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest before the crash");
        assert_eq!((crashed.sessions, crashed.tasks), (1, 0));

        let other = scratch.other_session(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let active = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest while the crashed session is recent");
        assert_eq!((active.sessions, active.tasks), (1, 0));
        assert!(!Scratch::ended(&ingest));

        scratch.go_idle();
        let swept = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest after the crashed session went idle");
        assert_eq!((swept.sessions, swept.tasks, swept.created), (2, 1, 1));
        assert!(Scratch::ended(&ingest));
        let titles: Vec<_> = ingest
            .store
            .list(None, true)
            .expect("procedures list")
            .into_iter()
            .map(|row| row.procedure.title)
            .collect();
        assert_eq!(
            titles,
            ["Page 2 in src/paginate.js repeats the last product from page 1"]
        );

        let again = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest with nothing idle");
        assert_eq!((again.sessions, again.tasks), (1, 0));
    }

    #[test]
    fn idle_sessions_are_left_alone_while_paused() {
        let scratch = Scratch::new("never-ended-paused");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest before the crash");
        scratch.go_idle();
        let other = scratch.other_session(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));

        ingest.store.set_paused(true).expect("capture pauses");
        let paused = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest while paused");
        assert_eq!(paused, IngestReport::default());
        assert!(!Scratch::ended(&ingest));

        ingest.store.set_paused(false).expect("capture resumes");
        let resumed = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest after resuming capture");
        assert_eq!((resumed.tasks, resumed.created), (1, 1));
        assert!(Scratch::ended(&ingest));
    }

    #[test]
    fn sessions_whose_transcript_is_gone_are_ended() {
        let scratch = Scratch::new("never-ended-gone");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .transcript(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("ingest before the crash");
        fs::remove_file(&scratch.transcript).expect("transcript is removed");

        let other = scratch.other_session(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let report = ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("ingest after the transcript is gone");
        assert_eq!((report.sessions, report.tasks), (1, 0));
        assert!(Scratch::ended(&ingest));
        assert_eq!(
            ingest
                .store
                .open_sessions(Harness::ClaudeCode.as_str())
                .expect("open sessions list"),
            [(
                OTHER_SESSION.to_owned(),
                other.to_string_lossy().into_owned()
            )]
        );
    }

    #[test]
    fn each_ingest_finishes_a_bounded_number_of_idle_sessions() {
        let scratch = Scratch::new("never-ended-bounded");
        let mut ingest = scratch.ingest();
        let stamp = Timestamp::now().to_string();
        for index in 0..=Ingest::IDLE_SESSIONS_PER_RUN {
            ingest
                .store
                .set_progress(
                    &format!("gone-{index}"),
                    Harness::ClaudeCode.as_str(),
                    &scratch
                        .dir
                        .join(format!("gone-{index}.jsonl"))
                        .to_string_lossy(),
                    None,
                    Progress {
                        extracted_through: None,
                        ended: false,
                    },
                    &stamp,
                )
                .expect("progress saves");
        }
        let other = scratch.other_session(&Scratch::task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));

        ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("first ingest");
        assert_eq!(
            ingest
                .store
                .open_sessions(Harness::ClaudeCode.as_str())
                .expect("open sessions list")
                .len(),
            2
        );
        ingest
            .transcript(Harness::ClaudeCode, &other, false)
            .expect("second ingest");
        assert_eq!(
            ingest
                .store
                .open_sessions(Harness::ClaudeCode.as_str())
                .expect("open sessions list")
                .len(),
            1
        );
    }

    #[test]
    fn sessions_are_attributed_to_the_commit_checked_out_while_they_ran() {
        let cases: [(&str, &[&str], Option<usize>); 3] = [
            (
                "commit-before",
                &["@1790845200 +0000", "@1790852400 +0000"],
                Some(0),
            ),
            ("commit-after", &["@1790852400 +0000"], None),
            ("commit-none", &[], None),
        ];
        for (name, committed, expected) in cases {
            let scratch = Scratch::new(name);
            let (repo, commits) = scratch.repository(committed);
            let mut ingest = scratch.ingest();
            scratch.append_in(
                &repo.to_string_lossy(),
                &Scratch::task(
                    0,
                    "The cart total in src/cart.js ignores the discount code",
                    "src/cart.js",
                ),
            );
            let report = ingest
                .transcript(Harness::ClaudeCode, &scratch.transcript, true)
                .expect("ingest in a repository");
            assert_eq!(report.created, 1, "{name}");
            assert_eq!(
                Scratch::commits(&ingest),
                [expected.map(|index| commits[index].clone())],
                "{name}"
            );
        }
    }

    #[test]
    fn relative_transcripts_are_recorded_by_absolute_path() {
        let scratch = Scratch::new("relative-transcript");
        let mut ingest = scratch.ingest();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let relative: PathBuf = std::env::current_dir()
            .expect("current directory is known")
            .components()
            .skip(1)
            .map(|_| Path::new(".."))
            .chain(
                scratch
                    .transcript
                    .components()
                    .skip(1)
                    .map(|part| Path::new(part.as_os_str())),
            )
            .collect();
        assert!(relative.is_relative());
        ingest
            .transcript(Harness::ClaudeCode, &relative, false)
            .expect("ingest by relative path");

        let open = ingest
            .store
            .open_sessions(Harness::ClaudeCode.as_str())
            .expect("open sessions list");
        let [(session, recorded)] = open.as_slice() else {
            panic!("one open session: {open:?}");
        };
        assert_eq!(session, SESSION);
        assert!(Path::new(recorded).is_absolute(), "{recorded}");
        assert_eq!(
            fs::canonicalize(recorded).expect("recorded transcript exists"),
            fs::canonicalize(&scratch.transcript).expect("transcript exists")
        );
    }

    #[test]
    fn background_ingests_leave_their_transcript_to_the_running_one() {
        let scratch = Scratch::new("deferred");
        let home = scratch.home();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let running = scratch.ingest();

        let deferred = Ingest::run_or_defer(&home, Harness::ClaudeCode, &scratch.transcript, true)
            .expect("background ingest defers");
        assert_eq!(deferred, None);
        assert!(PendingIngests::new(&home).any().expect("markers list"));

        let report = running.finish().expect("running ingest finishes");
        assert_eq!((report.sessions, report.tasks, report.created), (1, 1, 1));
        assert!(!PendingIngests::new(&home).any().expect("markers list"));
        let ingest = scratch.ingest();
        assert!(Scratch::ended(&ingest));
    }

    #[test]
    fn transcripts_queued_as_the_running_ingest_releases_are_picked_up() {
        let scratch = Scratch::new("deferred-race");
        let home = scratch.home();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let mut running = scratch.ingest();
        assert_eq!(
            running.drain().expect("nothing is queued yet"),
            IngestReport::default()
        );
        assert_eq!(
            Ingest::run_or_defer(&home, Harness::ClaudeCode, &scratch.transcript, true)
                .expect("background ingest defers"),
            None
        );
        drop(running);

        let report = Ingest::pick_up(&home).expect("queued transcripts are picked up");
        assert_eq!((report.sessions, report.created), (1, 1));
        assert!(!PendingIngests::new(&home).any().expect("markers list"));
    }

    #[test]
    fn queued_transcripts_are_ingested_once_and_remember_ending() {
        let scratch = Scratch::new("deferred-merge");
        let home = scratch.home();
        let other = scratch.dir.join("other.jsonl");
        let pending = PendingIngests::new(&home);
        pending
            .push(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("marker queues");
        pending
            .push(Harness::ClaudeCode, &scratch.transcript, true)
            .expect("marker queues");
        pending
            .push(Harness::ClaudeCode, &other, false)
            .expect("marker queues");
        pending
            .push(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("marker queues");

        let mut taken = pending.take().expect("markers are taken");
        taken.sort_by(|left, right| left.transcript.cmp(&right.transcript));
        assert_eq!(
            taken,
            [
                PendingIngest {
                    harness: Harness::ClaudeCode.as_str().to_owned(),
                    transcript: other,
                    ended: false
                },
                PendingIngest {
                    harness: Harness::ClaudeCode.as_str().to_owned(),
                    transcript: scratch.transcript.clone(),
                    ended: true
                },
            ]
        );
        assert!(!pending.any().expect("markers list"));
        assert_eq!(pending.take().expect("markers are taken"), []);
    }

    #[test]
    fn queued_transcripts_of_an_unknown_harness_are_skipped() {
        let scratch = Scratch::new("deferred-harness");
        let home = scratch.home();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let queued = home.pending_ingests();
        fs::create_dir_all(&queued).expect("queue directory is writable");
        fs::write(
            queued.join("1-1-0.marker"),
            json!({"harness": "future-agent", "transcript": scratch.transcript, "ended": true})
                .to_string(),
        )
        .expect("marker is written");

        let report = scratch.ingest().finish().expect("running ingest finishes");

        assert_eq!(report, IngestReport::default());
        assert!(!PendingIngests::new(&home).any().expect("markers list"));
        assert_eq!(
            Scratch::logged(&home),
            [format!(
                "ingest {}: unsupported harness `future-agent`",
                scratch.transcript.display()
            )]
        );
    }

    #[test]
    fn unreadable_queued_markers_are_logged_and_dropped() {
        let scratch = Scratch::new("deferred-corrupt");
        let home = scratch.home();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let queued = home.pending_ingests();
        fs::create_dir_all(&queued).expect("queue directory is writable");
        let corrupt = queued.join("1-1-0.marker");
        fs::write(&corrupt, r#"{"harness":"claude-code","transcr"#).expect("marker is written");
        fs::write(
            queued.join("1-1-1.marker"),
            json!({"harness": "claude-code", "transcript": scratch.transcript, "ended": true})
                .to_string(),
        )
        .expect("marker is written");

        let report = scratch.ingest().finish().expect("running ingest finishes");

        assert_eq!((report.sessions, report.created), (1, 1));
        assert!(!PendingIngests::new(&home).any().expect("markers list"));
        let logged = Scratch::logged(&home);
        assert_eq!(logged.len(), 1, "{logged:?}");
        assert!(
            logged[0].starts_with(&format!("parse the queued ingest {}: ", corrupt.display())),
            "{logged:?}"
        );
    }

    #[test]
    fn queued_transcripts_that_fail_are_logged() {
        let scratch = Scratch::new("deferred-failures");
        let home = scratch.home();
        scratch.append_in(
            "shop",
            &Scratch::task(
                0,
                "Page 2 in src/paginate.js repeats the last product from page 1",
                "src/paginate.js",
            ),
        );
        let missing = scratch.dir.join("missing.jsonl");
        let healthy = scratch.other_session(&Scratch::task(
            0,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));

        for transcript in [&scratch.transcript, &missing, &healthy] {
            Ingest::run_or_defer(&home, Harness::ClaudeCode, transcript, true)
                .expect("background ingest runs");
        }

        let logged = Scratch::logged(&home);
        assert_eq!(logged.len(), 2, "{logged:?}");
        assert_eq!(
            logged[0],
            format!(
                "ingest {}: the transcript has no usable working directory",
                scratch.transcript.display()
            )
        );
        assert!(
            logged[1].starts_with(&format!(
                "ingest {}: read the transcript: ",
                missing.display()
            )),
            "{logged:?}"
        );
        assert_eq!(
            logged[1].matches(&*missing.to_string_lossy()).count(),
            1,
            "{logged:?}"
        );
        let ingest = scratch.ingest();
        assert!(
            ingest
                .store
                .progress(OTHER_SESSION)
                .expect("progress reads")
                .is_some_and(|progress| progress.ended)
        );
    }

    #[test]
    fn queued_markers_name_their_harness_by_its_stored_id() {
        let scratch = Scratch::new("deferred-format");
        let home = scratch.home();
        scratch.append(&Scratch::task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let queued = home.pending_ingests();
        fs::create_dir_all(&queued).expect("queue directory is writable");
        fs::write(
            queued.join("1-1-0.marker"),
            json!({"harness": "claude-code", "transcript": scratch.transcript, "ended": true})
                .to_string(),
        )
        .expect("marker is written");

        let report = scratch.ingest().finish().expect("running ingest finishes");

        assert_eq!((report.sessions, report.created), (1, 1));
        PendingIngests::new(&home)
            .push(Harness::ClaudeCode, &scratch.transcript, false)
            .expect("marker queues");
        let marker = PendingIngests::new(&home)
            .markers()
            .expect("markers list")
            .remove(0);
        let written: Value = serde_json::from_slice(&fs::read(marker).expect("marker reads"))
            .expect("marker is JSON");
        assert_eq!(written["harness"], "claude-code");
    }

    #[test]
    fn concurrent_background_ingests_learn_every_session() {
        let scratch = Scratch::new("deferred-concurrent");
        let home = scratch.home();
        let sessions: Vec<_> = (0..8)
            .map(|index| scratch.numbered_session(index))
            .collect();

        let workers: Vec<_> = sessions
            .iter()
            .map(|(_, transcript)| {
                let home = home.clone();
                let transcript = transcript.clone();
                std::thread::spawn(move || {
                    Ingest::run_or_defer(&home, Harness::ClaudeCode, &transcript, true)
                })
            })
            .collect();
        for worker in workers {
            worker
                .join()
                .expect("worker finishes")
                .expect("background ingest succeeds");
        }

        assert!(!PendingIngests::new(&home).any().expect("markers list"));
        let ingest = scratch.ingest();
        for (session, _) in &sessions {
            let progress = ingest
                .store
                .progress(session)
                .expect("progress reads")
                .expect("session is recorded");
            assert!(progress.ended, "{session}");
        }
        assert_eq!(
            ingest
                .store
                .list(None, true)
                .expect("procedures list")
                .len(),
            sessions.len()
        );
    }

    #[test]
    fn a_journal_is_removed_once_its_session_has_ended() {
        let scratch = Scratch::new("journal-removed");
        let home = scratch.home();
        let journal = home.journal(Harness::Cursor.as_str(), "conv-1");
        fs::create_dir_all(journal.parent().expect("journals have a directory"))
            .expect("journal directory is writable");
        let redactor = trodden_redact::Redactor::new();
        let observer = trodden_capture::journal::Observer {
            session: "conv-1",
            cwd: &scratch.dir.to_string_lossy(),
            at: Timestamp::now(),
            model: None,
            redactor: &redactor,
        };
        fs::write(
            &journal,
            observer
                .prompt("Fix the paging bug")
                .to_line()
                .expect("line encodes"),
        )
        .expect("journal is writable");
        let mut ingest = Ingest::start(&home).expect("ingest starts");

        ingest
            .transcript(Harness::Cursor, &journal, false)
            .expect("an open session ingests");
        assert!(journal.exists(), "an open session keeps its journal");
        ingest
            .transcript(Harness::Cursor, &journal, true)
            .expect("an ended session ingests");

        assert!(!journal.exists(), "an ended session's journal is removed");
        assert_eq!(
            ingest.store.progress("conv-1").expect("progress reads"),
            None,
            "a resumed conversation starts over"
        );
    }
}
