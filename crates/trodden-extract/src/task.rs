use std::sync::LazyLock;

use regex::Regex;
use trodden_core::{
    Trace,
    trace::{Event, EventKind},
};

const FOLLOW_UP_MAX_WORDS: usize = 3;

static FOLLOW_UP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?ix)^\s*(?:
            yes|yep|yeah|ok|okay|sure|go\ ahead|continue|proceed|keep\ going|
            try\ again|retry|again|do\ it|please\ do|still|
            that\ (?:didn't|did\ not|doesn't|does\ not|broke)|it\ (?:didn't|still|fails|is\ failing|broke)|
            (?:it's|its)\ (?:still|failing|broken)|not\ working|doesn't\ work|
            fix\ (?:it|that|this)|wrong|no|why
        )\b",
    )
    .expect("follow-up pattern is valid")
});

#[derive(Debug, Clone, Copy)]
pub(crate) struct Task<'a> {
    pub(crate) summary: &'a str,
    pub(crate) events: &'a [Event],
}

impl<'a> Task<'a> {
    pub(crate) fn split(trace: &'a Trace) -> Vec<Self> {
        let mut starts: Vec<(usize, &str)> = Vec::new();
        for (index, event) in trace.events.iter().enumerate() {
            if let EventKind::Prompt { summary } = &event.kind
                && (starts.is_empty() || !Self::is_follow_up(summary))
            {
                starts.push((index, summary));
            }
        }
        starts
            .iter()
            .enumerate()
            .map(|(i, &(start, summary))| {
                let end = starts
                    .get(i + 1)
                    .map_or(trace.events.len(), |&(next, _)| next);
                Self {
                    summary,
                    events: &trace.events[start..end],
                }
            })
            .collect()
    }

    fn is_follow_up(summary: &str) -> bool {
        summary.split_whitespace().count() <= FOLLOW_UP_MAX_WORDS || FOLLOW_UP.is_match(summary)
    }

    pub(crate) fn first_seq(&self) -> u32 {
        self.events.first().map_or(0, |event| event.seq)
    }

    pub(crate) fn last_seq(&self) -> u32 {
        self.events.last().map_or(0, |event| event.seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_follow_ups() {
        for prompt in [
            "try again",
            "yes",
            "that didn't work, the test still fails",
            "Still failing on CI",
        ] {
            assert!(Task::is_follow_up(prompt), "{prompt}");
        }
        for prompt in [
            "Add a `count` subcommand that prints the number of invoices.",
            "Nowhere in the docs is the API described",
        ] {
            assert!(!Task::is_follow_up(prompt), "{prompt}");
        }
    }
}
