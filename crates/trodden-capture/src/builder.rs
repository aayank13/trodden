use std::collections::HashMap;

use anyhow::{Result, bail};
use jiff::Timestamp;
use trodden_core::{
    HarnessId, SessionId, Trace,
    trace::{Event, EventKind, FileChange, ToolAction, ToolArgs, ToolCall, ToolOutcome},
};
use trodden_redact::Redactor;

use crate::{
    ErrorSignature,
    command::Command,
    diff::Diff,
    evidence::CheckOutput,
    symbols::{Hunk, SymbolFinder},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Finish {
    Succeeded,
    Failed { exit_code: Option<i32> },
    Interrupted,
}

#[derive(Debug)]
pub struct Reminder;

impl Reminder {
    pub const PREFIX: &'static str = "Trodden: the procedure recalled";
}

#[derive(Debug)]
pub(crate) struct Prompts {
    pub(crate) skipped: &'static [&'static str],
    pub(crate) pasted: Option<&'static str>,
}

pub(crate) struct TraceBuilder<'a> {
    harness: &'static str,
    redactor: &'a Redactor,
    session: Option<String>,
    cwd: String,
    here: String,
    model: Option<String>,
    started_at: Option<Timestamp>,
    now: Timestamp,
    events: Vec<Event>,
    pending: HashMap<String, usize>,
}

impl<'a> TraceBuilder<'a> {
    pub(crate) const SUMMARY_CHARS: usize = 200;

    pub(crate) const SUMMARY_SOURCE_BYTES: usize = 64 * 1024;

    pub(crate) const REDACTED: &'static str = "[REDACTED:";

    const SEPARATORS: &'static [char] = &['/', '\\'];

    pub(crate) fn new(harness: &'static str, redactor: &'a Redactor) -> Self {
        Self {
            harness,
            redactor,
            session: None,
            cwd: String::new(),
            here: String::new(),
            model: None,
            started_at: None,
            now: Timestamp::UNIX_EPOCH,
            events: Vec::new(),
            pending: HashMap::new(),
        }
    }

    pub(crate) fn session(&mut self, id: &str) {
        if self.session.is_none() && !id.is_empty() {
            self.session = Some(id.to_owned());
        }
    }

    pub(crate) fn directory(&mut self, cwd: &str) {
        let cwd = cwd.trim_end_matches(Self::SEPARATORS);
        if cwd.is_empty() {
            return;
        }
        if self.cwd.is_empty() {
            cwd.clone_into(&mut self.cwd);
        }
        cwd.clone_into(&mut self.here);
    }

    pub(crate) fn working_directory(&self) -> &str {
        &self.cwd
    }

    pub(crate) fn time(&mut self, at: &str) {
        if let Ok(at) = at.parse::<Timestamp>() {
            self.time_at(at);
        }
    }

    pub(crate) fn time_at(&mut self, at: Timestamp) {
        self.now = at;
        self.started_at.get_or_insert(at);
    }

    pub(crate) fn model(&mut self, model: &str) {
        if self.model.is_none() && !model.is_empty() && !model.starts_with('<') {
            self.model = Some(model.to_owned());
        }
    }

    pub(crate) fn prompt(&mut self, text: &str, prompts: &Prompts) {
        if let Some(summary) = Self::summarize(text, prompts, self.redactor) {
            self.emit(EventKind::Prompt { summary });
        }
    }

    pub(crate) fn summarize(text: &str, prompts: &Prompts, redactor: &Redactor) -> Option<String> {
        let text = text.trim();
        if text.is_empty()
            || text.starts_with(Reminder::PREFIX)
            || prompts
                .skipped
                .iter()
                .any(|prefix| text.starts_with(prefix))
        {
            return None;
        }
        let headline = Self::headline(text, prompts);
        (!headline.is_empty()).then(|| Self::summary(headline, redactor))
    }

    fn headline<'t>(text: &'t str, prompts: &Prompts) -> &'t str {
        let mut wrapper: Option<&str> = None;
        let mut pasted = None;
        for line in text.lines() {
            let mut rest = line.trim();
            while !rest.is_empty() {
                if let Some(tag) = wrapper {
                    let Some(close) = rest.find(&format!("</{}", &tag[1..])) else {
                        if Some(tag) == prompts.pasted {
                            pasted.get_or_insert(rest);
                        }
                        break;
                    };
                    let inside = rest[..close].trim();
                    if Some(tag) == prompts.pasted && !inside.is_empty() {
                        pasted.get_or_insert(inside);
                    }
                    wrapper = None;
                    rest = Self::after_tag(&rest[close..]);
                } else if let Some(tag) = Self::wrapper(rest, prompts) {
                    wrapper = Some(tag);
                    rest = Self::after_tag(rest);
                } else {
                    return rest;
                }
            }
        }
        pasted.unwrap_or_default()
    }

    fn wrapper(line: &str, prompts: &Prompts) -> Option<&'static str> {
        prompts
            .skipped
            .iter()
            .chain(prompts.pasted.as_ref())
            .map(|prefix| prefix.trim_end_matches('>'))
            .filter(|tag| tag.starts_with('<'))
            .find(|tag| line.starts_with(tag))
    }

    fn after_tag(text: &str) -> &str {
        text.find('>').map_or("", |end| text[end + 1..].trim())
    }

    fn summary(line: &str, redactor: &Redactor) -> String {
        let source = if line.len() > Self::SUMMARY_SOURCE_BYTES {
            line[..line.floor_char_boundary(Self::SUMMARY_SOURCE_BYTES)]
                .trim_end_matches(|c: char| !c.is_whitespace())
        } else {
            line
        };
        let redacted = redactor.redact(source);
        Self::truncate(&redacted, Self::SUMMARY_CHARS)
            .trim_end()
            .to_owned()
    }

    fn truncate(text: &str, chars: usize) -> &str {
        let Some((cut, _)) = text.char_indices().nth(chars) else {
            return text;
        };
        let straddling = text
            .match_indices(Self::REDACTED)
            .map(|(start, _)| start)
            .take_while(|&start| start < cut)
            .last()
            .filter(|&start| text[start..].find(']').is_none_or(|end| start + end >= cut));
        &text[..straddling.unwrap_or(cut)]
    }

    pub(crate) fn shell(&self, command: &str) -> (ToolArgs, ToolAction) {
        let subdirectory = self.subdirectory();
        let base = if subdirectory.is_some() {
            &self.here
        } else {
            &self.cwd
        };
        let command = Command::normalize(command, base);
        let text = match subdirectory {
            Some(dir) if !command.text().is_empty() => {
                format!("cd {} && {}", Self::quoted(dir), command.text())
            }
            _ => command.text().to_owned(),
        };
        let args = ToolArgs {
            command: Some(self.redactor.redact(&text).into_owned()),
            ..ToolArgs::default()
        };
        (args, command.action())
    }

    pub(crate) fn redact(&self, text: &str) -> String {
        self.redactor.redact(text).into_owned()
    }

    pub(crate) fn search_pattern(&self, pattern: &str, glob: bool) -> String {
        if glob && Self::is_absolute(pattern) {
            self.relative(pattern)
        } else {
            self.redact(pattern)
        }
    }

    pub(crate) fn call(
        &mut self,
        id: Option<&str>,
        tool: &str,
        action: ToolAction,
        args: ToolArgs,
    ) -> usize {
        let index = self.emit(EventKind::ToolCall(ToolCall {
            tool: tool.to_owned(),
            action,
            args,
            outcome: ToolOutcome::Interrupted,
            changes: Vec::new(),
            duration_ms: None,
            error: None,
            confirmed_by_output: false,
        }));
        if let Some(id) = id {
            self.pending.insert(id.to_owned(), index);
        }
        index
    }

    pub(crate) fn is_pending(&self, id: &str) -> bool {
        self.pending.contains_key(id)
    }

    pub(crate) fn pending_call(&self, id: &str) -> Option<&ToolCall> {
        let index = *self.pending.get(id)?;
        match &self.events[index].kind {
            EventKind::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    // A masked exit code (`| tail`, `|| true`) leaves the outcome to the check's own summary
    // line.
    pub(crate) fn resolve(
        &mut self,
        id: &str,
        finish: Finish,
        output: &str,
        duration_ms: Option<u64>,
        changes: Vec<FileChange>,
    ) {
        let Some(index) = self.pending.remove(id) else {
            return;
        };
        let redactor = self.redactor;
        let EventKind::ToolCall(call) = &mut self.events[index].kind else {
            return;
        };
        let verdict = call
            .args
            .command
            .as_deref()
            .and_then(|command| CheckOutput::verdict(command, output));
        call.outcome = match (finish, verdict) {
            (Finish::Interrupted, _) => ToolOutcome::Interrupted,
            (Finish::Failed { .. }, Some(true)) => ToolOutcome::Succeeded,
            (Finish::Failed { exit_code }, _) => ToolOutcome::Failed { exit_code },
            (Finish::Succeeded, Some(false)) => ToolOutcome::Failed { exit_code: None },
            (Finish::Succeeded, _) => ToolOutcome::Succeeded,
        };
        call.confirmed_by_output = verdict.is_some() && call.outcome != ToolOutcome::Interrupted;
        if matches!(call.outcome, ToolOutcome::Failed { .. }) {
            call.error = ErrorSignature::of(output, redactor);
        }
        call.duration_ms = duration_ms;
        if call.action == ToolAction::Edit && !matches!(finish, Finish::Failed { .. }) {
            call.changes = changes;
        }
    }

    pub(crate) fn change(
        &self,
        path: &str,
        created: bool,
        diff: &Diff,
        original: Option<&str>,
    ) -> FileChange {
        let hunks: Vec<Hunk> = diff
            .starts
            .iter()
            .map(|&old_start| Hunk { old_start })
            .collect();
        FileChange {
            path: self.relative(path),
            created,
            symbols: SymbolFinder::find(diff.changed.iter().map(String::as_str), original, &hunks),
            lines_added: diff.added,
            lines_removed: diff.removed,
        }
    }

    pub(crate) fn compaction(&mut self, automatic: bool) {
        self.emit(EventKind::Compaction { automatic });
    }

    pub(crate) fn emit(&mut self, kind: EventKind) -> usize {
        let seq = u32::try_from(self.events.len()).unwrap_or(u32::MAX);
        self.events.push(Event {
            seq,
            at: self.now,
            kind,
        });
        self.events.len() - 1
    }

    pub(crate) fn finish(self, missing: &str) -> Result<Trace> {
        let Some(session) = self.session else {
            bail!("{missing}");
        };
        Ok(Trace {
            session: SessionId::new(session),
            harness: HarnessId::new(self.harness),
            model: self.model,
            cwd: self.redactor.redact(&self.cwd).into_owned(),
            commit: None,
            started_at: self.started_at.unwrap_or(self.now),
            events: self.events,
        })
    }

    pub(crate) fn relative(&self, path: &str) -> String {
        let (base, rest) = if Self::is_absolute(path) {
            match self.below(path) {
                Some(rest) => ("", rest),
                None => return self.redactor.redact(path).into_owned(),
            }
        } else {
            (self.subdirectory().unwrap_or_default(), path)
        };
        let separators = self.separators();
        let parts = Self::normalized(base.split(separators).chain(rest.split(separators)));
        let text = if parts.first() == Some(&"..") && !self.cwd.is_empty() {
            let outside = self.outside(&parts);
            match self.below(&outside) {
                Some(rest) => Self::joined(&Self::normalized(rest.split(separators))),
                None => outside,
            }
        } else {
            Self::joined(&parts)
        };
        self.redactor.redact(&text).into_owned()
    }

    fn joined(parts: &[&str]) -> String {
        if parts.is_empty() {
            ".".to_owned()
        } else {
            parts.join("/")
        }
    }

    fn outside(&self, parts: &[&str]) -> String {
        let root = Self::root(&self.cwd);
        let separator = if self.cwd.contains('\\') { "\\" } else { "/" };
        let resolved = Self::normalized(
            self.cwd[root.len()..]
                .split(self.separators())
                .chain(parts.iter().copied()),
        );
        let below_root: Vec<&str> = resolved
            .into_iter()
            .skip_while(|part| *part == "..")
            .collect();
        format!("{root}{}", below_root.join(separator))
    }

    fn below<'p>(&self, path: &'p str) -> Option<&'p str> {
        if self.cwd.is_empty() {
            return None;
        }
        let head = path.get(..self.cwd.len())?;
        let same = if self.is_windows() {
            head.chars().zip(self.cwd.chars()).all(|(a, b)| {
                a.eq_ignore_ascii_case(&b)
                    || (Self::SEPARATORS.contains(&a) && Self::SEPARATORS.contains(&b))
            })
        } else {
            head == self.cwd
        };
        let rest = &path[self.cwd.len()..];
        (same && (rest.is_empty() || rest.starts_with(self.separators())))
            .then(|| rest.trim_start_matches(self.separators()))
    }

    fn subdirectory(&self) -> Option<&str> {
        self.below(&self.here).filter(|dir| !dir.is_empty())
    }

    fn is_windows(&self) -> bool {
        Self::has_drive(&self.cwd) || self.cwd.starts_with(r"\\")
    }

    fn separators(&self) -> &'static [char] {
        if self.is_windows() {
            Self::SEPARATORS
        } else {
            &Self::SEPARATORS[..1]
        }
    }

    pub(crate) fn is_absolute(path: &str) -> bool {
        path.starts_with(['~', '/', '\\'])
            || (Self::has_drive(path) && path[2..].starts_with(Self::SEPARATORS))
    }

    fn has_drive(path: &str) -> bool {
        matches!(path.as_bytes(), [letter, b':', ..] if letter.is_ascii_alphabetic())
    }

    fn root(path: &str) -> &str {
        let drive = if Self::has_drive(path) { 2 } else { 0 };
        let rest = path[drive..].trim_start_matches(Self::SEPARATORS);
        &path[..path.len() - rest.len()]
    }

    fn normalized<'p>(parts: impl IntoIterator<Item = &'p str>) -> Vec<&'p str> {
        let mut normalized = Vec::new();
        for part in parts {
            match part {
                "" | "." => {}
                ".." if normalized.last().is_some_and(|last| *last != "..") => {
                    normalized.pop();
                }
                _ => normalized.push(part),
            }
        }
        normalized
    }

    fn quoted(dir: &str) -> String {
        if dir
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "._-/+@%,:=".contains(c))
        {
            dir.to_owned()
        } else {
            format!("'{}'", dir.replace('\'', r"'\''"))
        }
    }
}
