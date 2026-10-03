use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{self, Path, PathBuf},
};

use anyhow::{Context, Result};
use jiff::Timestamp;
use serde::Deserialize;
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

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TranscriptLine {
    cwd: Option<PathBuf>,
    #[serde(default)]
    is_sidechain: bool,
}

impl TranscriptLine {
    fn working_directory(text: &str) -> Option<PathBuf> {
        text.lines()
            .filter_map(|line| serde_json::from_str::<Self>(line).ok())
            .filter(|line| !line.is_sidechain)
            .find_map(|line| line.cwd)
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
        let mut report = self.session(transcript, ended)?;
        report.absorb(self.finish_idle_sessions()?);
        Ok(report)
    }

    fn finish_idle_sessions(&mut self) -> Result<IngestReport> {
        let mut report = IngestReport::default();
        if self.store.paused()? {
            return Ok(report);
        }
        let idle: Vec<_> = self
            .store
            .open_sessions(HARNESS)?
            .into_iter()
            .map(|(session, transcript)| (session, PathBuf::from(transcript)))
            .filter(|(_, transcript)| Self::is_idle(transcript) || !transcript.exists())
            .take(Self::IDLE_SESSIONS_PER_RUN)
            .collect();
        for (session, transcript) in idle {
            match self.session(&transcript, true) {
                Ok(finished) => report.absorb(finished),
                Err(_) => self
                    .store
                    .end_session(&session, &Timestamp::now().to_string())?,
            }
        }
        Ok(report)
    }

    fn session(&mut self, transcript: &Path, ended: bool) -> Result<IngestReport> {
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

        let cwd = TranscriptLine::working_directory(&text)
            .filter(|cwd| cwd.is_absolute())
            .with_context(|| format!("{} has no usable working directory", transcript.display()))?;
        let workspace = Workspace::resolve(&cwd, &self.store)?;
        let recorded = path::absolute(transcript)
            .with_context(|| format!("resolve {}", transcript.display()))?;
        let recorded = recorded.to_string_lossy();
        let now = Timestamp::now();
        let stamp = now.to_string();
        if self.store.paused()? {
            self.store.set_progress(
                &session,
                HARNESS,
                &recorded,
                Some(workspace.repo.as_str()),
                Progress {
                    extracted_through: last_seq.max(already),
                    ended,
                },
                &stamp,
            )?;
            return Ok(IngestReport::default());
        }
        trace.commit = workspace.head();
        let mut report = IngestReport {
            sessions: 1,
            ..IngestReport::default()
        };
        let mut extracted_through = already;
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
        Ok(report)
    }

    pub fn backfill_claude_code(
        &mut self,
        projects: &Path,
    ) -> Result<(IngestReport, Vec<(PathBuf, String)>)> {
        if self.store.paused()? {
            return Ok((IngestReport::default(), Vec::new()));
        }
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
            match self.session(&transcript, false) {
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
    use std::io::Write;

    use serde_json::{Value, json};
    use trodden_store::{Cue, Injection};

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

        fn stored(&self) -> String {
            ["trodden.db", "trodden.db-wal"]
                .iter()
                .map(|name| fs::read(self.dir.join("home").join(name)).unwrap_or_default())
                .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
                .collect()
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

    #[test]
    fn work_done_while_paused_is_never_learned() {
        let scratch = Scratch::new("paused");
        let mut ingest = scratch.ingest();
        ingest.store.set_paused(true).expect("capture pauses");
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let paused = ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest while paused");
        assert_eq!(paused, IngestReport::default());

        ingest.store.set_paused(false).expect("capture resumes");
        let ended = ingest
            .claude_code(&scratch.transcript, true)
            .expect("ingest after resuming capture");
        assert_eq!((ended.sessions, ended.tasks, ended.created), (1, 0, 0));

        scratch.append(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let resumed = ingest
            .claude_code(&scratch.transcript, true)
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
    fn sessions_spanning_a_pause_learn_only_what_followed_it() {
        let scratch = Scratch::new("spanning-pause");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .claude_code(&scratch.transcript, true)
            .expect("first ingest");

        scratch.append(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        inject(&ingest, 5);
        ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest before pausing");

        ingest.store.set_paused(true).expect("capture pauses");
        scratch.append(&task(
            10,
            "Search in src/search.js returns archived products to shoppers",
            "src/search.js",
        ));
        ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest while paused");

        ingest.store.set_paused(false).expect("capture resumes");
        scratch.append(&task(15, "still failing", "src/search.js"));
        inject(&ingest, 15);
        scratch.append(&task(
            20,
            "Checkout in src/checkout.js charges shipping twice",
            "src/checkout.js",
        ));
        let resumed = ingest
            .claude_code(&scratch.transcript, true)
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
                &task(
                    0,
                    &format!("The cart total in {cwd}/src/cart.js ignores the discount code"),
                    "src/cart.js",
                ),
            );
            let report = ingest
                .claude_code(&scratch.transcript, true)
                .expect("ingest in an unusual directory");
            assert_eq!((report.sessions, report.tasks, report.created), (1, 1, 1));
            assert!(!scratch.stored().contains(email), "{cwd}");
        }
    }

    #[test]
    fn sessions_without_an_absolute_directory_are_refused() {
        let scratch = Scratch::new("relative-cwd");
        let mut ingest = scratch.ingest();
        scratch.append_in("shop", &task(0, "Fix the cart total", "src/cart.js"));
        let error = ingest
            .claude_code(&scratch.transcript, true)
            .expect_err("a relative directory is refused");
        assert!(
            error
                .to_string()
                .ends_with("has no usable working directory"),
            "{error}"
        );
    }

    #[test]
    fn backfill_while_paused_leaves_history_for_later() {
        let scratch = Scratch::new("paused-backfill");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        scratch.append(&task(
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
            .backfill_claude_code(&projects)
            .expect("backfill while paused");
        assert_eq!(paused, IngestReport::default());
        assert_eq!(
            ingest.store.progress(SESSION).expect("progress reads"),
            None
        );

        ingest.store.set_paused(false).expect("capture resumes");
        let (resumed, failures) = ingest
            .backfill_claude_code(&projects)
            .expect("backfill after resuming capture");
        assert!(failures.is_empty());
        assert_eq!((resumed.sessions, resumed.created), (1, 1));
    }

    #[test]
    fn idle_sessions_that_never_ended_are_finished_by_later_ingests() {
        let scratch = Scratch::new("never-ended");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        let crashed = ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest before the crash");
        assert_eq!((crashed.sessions, crashed.tasks), (1, 0));

        let other = scratch.other_session(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let active = ingest
            .claude_code(&other, false)
            .expect("ingest while the crashed session is recent");
        assert_eq!((active.sessions, active.tasks), (1, 0));
        assert!(!Scratch::ended(&ingest));

        scratch.go_idle();
        let swept = ingest
            .claude_code(&other, false)
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
            .claude_code(&other, false)
            .expect("ingest with nothing idle");
        assert_eq!((again.sessions, again.tasks), (1, 0));
    }

    #[test]
    fn idle_sessions_are_left_alone_while_paused() {
        let scratch = Scratch::new("never-ended-paused");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest before the crash");
        scratch.go_idle();
        let other = scratch.other_session(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));

        ingest.store.set_paused(true).expect("capture pauses");
        let paused = ingest
            .claude_code(&other, false)
            .expect("ingest while paused");
        assert_eq!(paused, IngestReport::default());
        assert!(!Scratch::ended(&ingest));

        ingest.store.set_paused(false).expect("capture resumes");
        let resumed = ingest
            .claude_code(&other, false)
            .expect("ingest after resuming capture");
        assert_eq!((resumed.tasks, resumed.created), (1, 1));
        assert!(Scratch::ended(&ingest));
    }

    #[test]
    fn sessions_whose_transcript_is_gone_are_ended() {
        let scratch = Scratch::new("never-ended-gone");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
            0,
            "Page 2 in src/paginate.js repeats the last product from page 1",
            "src/paginate.js",
        ));
        ingest
            .claude_code(&scratch.transcript, false)
            .expect("ingest before the crash");
        fs::remove_file(&scratch.transcript).expect("transcript is removed");

        let other = scratch.other_session(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));
        let report = ingest
            .claude_code(&other, false)
            .expect("ingest after the transcript is gone");
        assert_eq!((report.sessions, report.tasks), (1, 0));
        assert!(Scratch::ended(&ingest));
        assert_eq!(
            ingest
                .store
                .open_sessions(HARNESS)
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
                    HARNESS,
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
        let other = scratch.other_session(&task(
            5,
            "The cart total in src/cart.js ignores the discount code",
            "src/cart.js",
        ));

        ingest.claude_code(&other, false).expect("first ingest");
        assert_eq!(
            ingest
                .store
                .open_sessions(HARNESS)
                .expect("open sessions list")
                .len(),
            2
        );
        ingest.claude_code(&other, false).expect("second ingest");
        assert_eq!(
            ingest
                .store
                .open_sessions(HARNESS)
                .expect("open sessions list")
                .len(),
            1
        );
    }

    #[test]
    fn relative_transcripts_are_recorded_by_absolute_path() {
        let scratch = Scratch::new("relative-transcript");
        let mut ingest = scratch.ingest();
        scratch.append(&task(
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
            .claude_code(&relative, false)
            .expect("ingest by relative path");

        let open = ingest
            .store
            .open_sessions(HARNESS)
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
}
