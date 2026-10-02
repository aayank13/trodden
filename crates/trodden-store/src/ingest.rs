use anyhow::{Context, Result};
use rusqlite::{OptionalExtension, params};

use crate::Store;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Progress {
    pub extracted_through: Option<u32>,
    pub ended: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionRecord {
    pub session: String,
    pub first_seq: u32,
    pub summary: String,
    pub procedure: Option<i64>,
    pub rejection: Option<String>,
    pub outcome: Option<String>,
    pub tool_calls: Option<u32>,
    pub span: Option<(String, String)>,
    pub at: String,
}

impl Store {
    pub fn progress(&self, session: &str) -> Result<Option<Progress>> {
        self.conn
            .query_row(
                "SELECT extracted_through, ended FROM sessions WHERE session = ?1",
                [session],
                |row| {
                    Ok(Progress {
                        extracted_through: row.get(0)?,
                        ended: row.get(1)?,
                    })
                },
            )
            .optional()
            .context("read session progress")
    }

    pub fn set_progress(
        &self,
        session: &str,
        harness: &str,
        transcript: &str,
        repo: Option<&str>,
        progress: Progress,
        at: &str,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO sessions (session, harness, transcript, repo, extracted_through, ended, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                 ON CONFLICT (session) DO UPDATE SET
                    transcript = excluded.transcript, repo = excluded.repo,
                    extracted_through = excluded.extracted_through, ended = excluded.ended,
                    updated_at = excluded.updated_at",
                params![session, harness, transcript, repo, progress.extracted_through, progress.ended, at],
            )
            .context("save session progress")?;
        Ok(())
    }

    pub fn record_extraction(&self, record: &ExtractionRecord) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR IGNORE INTO extractions
                    (session, first_seq, summary, procedure, rejection, outcome, tool_calls, started_at, ended_at, at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
                params![
                    record.session,
                    record.first_seq,
                    record.summary,
                    record.procedure,
                    record.rejection,
                    record.outcome,
                    record.tool_calls,
                    record.span.as_ref().map(|(start, _)| start),
                    record.span.as_ref().map(|(_, end)| end),
                    record.at
                ],
            )
            .context("record an extraction")?;
        Ok(())
    }

    pub fn rejections(&self, limit: usize) -> Result<Vec<ExtractionRecord>> {
        let mut statement = self
            .conn
            .prepare(
                "SELECT session, first_seq, summary, procedure, rejection, outcome, tool_calls,
                    started_at, ended_at, at
                 FROM extractions WHERE rejection IS NOT NULL ORDER BY at DESC LIMIT ?1",
            )
            .context("prepare the rejection listing")?;
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        statement
            .query_map([limit], |row| {
                let started: Option<String> = row.get(7)?;
                let ended: Option<String> = row.get(8)?;
                Ok(ExtractionRecord {
                    session: row.get(0)?,
                    first_seq: row.get(1)?,
                    summary: row.get(2)?,
                    procedure: row.get(3)?,
                    rejection: row.get(4)?,
                    outcome: row.get(5)?,
                    tool_calls: row.get(6)?,
                    span: started.zip(ended),
                    at: row.get(9)?,
                })
            })
            .context("list rejections")?
            .collect::<Result<_, _>>()
            .context("read rejections")
    }
}
