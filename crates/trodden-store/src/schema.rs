use anyhow::{Context, Result};

use crate::Store;

const MIGRATIONS: &[&str] = &[
    r#"
    CREATE TABLE settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE procedures (
        rowid INTEGER PRIMARY KEY,
        id TEXT NOT NULL,
        revision INTEGER NOT NULL,
        family TEXT NOT NULL,
        repo TEXT NOT NULL,
        state TEXT NOT NULL,
        title TEXT NOT NULL,
        signature TEXT NOT NULL,
        document TEXT NOT NULL,
        embedding BLOB,
        updated_at TEXT NOT NULL,
        UNIQUE (id, revision)
    );
    CREATE INDEX procedures_family ON procedures (family);
    CREATE INDEX procedures_repo ON procedures (repo, state);

    CREATE VIRTUAL TABLE procedures_fts USING fts5 (
        title, body,
        tokenize = "unicode61 tokenchars '_-./'"
    );

    CREATE TABLE entities (
        key TEXT NOT NULL,
        kind TEXT NOT NULL,
        procedure INTEGER NOT NULL REFERENCES procedures (rowid) ON DELETE CASCADE
    );
    CREATE INDEX entities_key ON entities (key);
    CREATE INDEX entities_procedure ON entities (procedure);

    CREATE TABLE extractions (
        session TEXT NOT NULL,
        first_seq INTEGER NOT NULL,
        summary TEXT NOT NULL,
        procedure INTEGER REFERENCES procedures (rowid) ON DELETE SET NULL,
        rejection TEXT,
        at TEXT NOT NULL,
        PRIMARY KEY (session, first_seq)
    );

    CREATE TABLE sessions (
        session TEXT PRIMARY KEY,
        harness TEXT NOT NULL,
        transcript TEXT NOT NULL,
        repo TEXT,
        extracted_through INTEGER NOT NULL,
        ended INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE repos (
        root TEXT PRIMARY KEY,
        repo TEXT NOT NULL
    );

    CREATE TABLE injections (
        rowid INTEGER PRIMARY KEY,
        session TEXT NOT NULL,
        procedure TEXT NOT NULL,
        revision INTEGER NOT NULL,
        holdout INTEGER NOT NULL,
        at TEXT NOT NULL
    );
    CREATE INDEX injections_session ON injections (session, procedure);
"#,
    r"
    CREATE VIRTUAL TABLE procedures_vocab USING fts5vocab (procedures_fts, 'row');
    INSERT INTO procedures_fts (procedures_fts, rank) VALUES ('rank', 'bm25(2.0, 1.0)');
",
    r"
    CREATE TABLE sessions_rebuilt (
        session TEXT PRIMARY KEY,
        harness TEXT NOT NULL,
        transcript TEXT NOT NULL,
        repo TEXT,
        extracted_through INTEGER,
        ended INTEGER NOT NULL,
        updated_at TEXT NOT NULL
    );
    INSERT INTO sessions_rebuilt
        SELECT session, harness, transcript, repo, extracted_through, ended, updated_at FROM sessions;
    DROP TABLE sessions;
    ALTER TABLE sessions_rebuilt RENAME TO sessions;
",
    r"
    ALTER TABLE procedures ADD COLUMN used_at TEXT;
    UPDATE procedures SET used_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now');

    CREATE TABLE embeddings (
        procedure INTEGER NOT NULL REFERENCES procedures (rowid) ON DELETE CASCADE,
        embedding BLOB NOT NULL
    );
    CREATE INDEX embeddings_procedure ON embeddings (procedure);
    INSERT INTO embeddings (procedure, embedding)
        SELECT rowid, embedding FROM procedures WHERE embedding IS NOT NULL;
    ALTER TABLE procedures DROP COLUMN embedding;

    ALTER TABLE injections ADD COLUMN cue TEXT NOT NULL DEFAULT 'prompt';
    ALTER TABLE injections ADD COLUMN task INTEGER;
    ALTER TABLE injections ADD COLUMN outcome TEXT;
    CREATE INDEX injections_outcomes ON injections (procedure, holdout, outcome);

    ALTER TABLE extractions ADD COLUMN outcome TEXT;
    ALTER TABLE extractions ADD COLUMN tool_calls INTEGER;
    ALTER TABLE extractions ADD COLUMN started_at TEXT;
    ALTER TABLE extractions ADD COLUMN ended_at TEXT;
",
];

impl Store {
    pub(crate) fn migrate(&mut self) -> Result<()> {
        let version: i64 = self
            .conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .context("read the schema version")?;
        let applied = usize::try_from(version).context("read a valid schema version")?;
        for (index, migration) in MIGRATIONS.iter().enumerate().skip(applied) {
            let tx = self.conn.transaction().context("start a migration")?;
            tx.execute_batch(migration)
                .with_context(|| format!("migrate the schema to version {}", index + 1))?;
            let next = i64::try_from(index + 1).context("count schema versions")?;
            tx.pragma_update(None, "user_version", next)
                .context("record the schema version")?;
            tx.commit().context("commit a migration")?;
        }
        Ok(())
    }
}
