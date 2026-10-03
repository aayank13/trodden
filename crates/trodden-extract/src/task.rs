use std::sync::LazyLock;

use regex::Regex;
use trodden_core::{
    Trace,
    trace::{Event, EventKind},
};

const SEPARATORS: &str = r"[\s\pP\pS]*";

const ACKNOWLEDGEMENTS: &str = r"
    yes|yep|yeah|yea|yup|ya|y|ok|okay|okey|okie|k|kk|sure|alright|all\s+right|right|correct|exactly|
    agreed|i\s+agree|indeed|absolutely|definitely|of\s+course|sounds\s+good|looks\s+good|lgtm|
    great|perfect|cool|nice|good|fine|awesome|thanks|thank\s+you|thx|ty|please|pls|plz|please\s+do|
    go|go\s+ahead|go\s+for\s+it|go\s+on|continue|proceed|carry\s+on|keep\s+going|resume|ship\s+it|
    do\s+(?:it|that|so|this|both)|let['’]?s\s+(?:do\s+(?:it|that|this)|go|continue|proceed)|
    try\s+again|retry|again|one\s+more\s+time|same|nope|no|wrong|failing|failed|broken|
    fix\s+(?:it|that|this|them)|undo|revert|done|(?:it|that|this)\s+(?:works|worked)|
    why|what|huh|hmm*|well|so|and|but|then|also|now|still|either|too|anymore|yet|for\s+me
";

const BACK_REFERENCES: &str = r"
    still\s+(?:
        fail\w*|broken|break\w*|not|no|doesn['’]?t|does\s+not|don['’]?t|isn['’]?t|is\s+not|won['’]?t|
        can['’]?t|get\w*|see\w*|the\s+same|same|error\w*|crash\w*|happen\w*|there|wrong|red|hang\w*|
        tim\w*\s+out|nothing|empty|blank|show\w*|return\w*|throw\w*|panic\w*
    )|
    (?:it|that|this|which)(?:['’]?s|\s+is|\s+was)?\s+(?:
        still|not|didn['’]?t|did\s+not|doesn['’]?t|does\s+not|don['’]?t|isn['’]?t|wasn['’]?t|
        won['’]?t|will\s+not|can['’]?t|broke|breaks|broken|fail\w*|crash\w*|keeps|hangs|throws|
        errors?|erroring|returns|gives|shows|says|prints|wrong|worse|incorrect|made\s+(?:it|things)\s+worse
    )(?:\s+(?:work\w*|help\w*|fix\s+(?:it|anything)|change\s+anything|right|correct))?|
    (?:not|doesn['’]?t|does\s+not|didn['’]?t|did\s+not|isn['’]?t|is\s+not|still\s+not|nothing)\s+work\w*|
    (?:didn['’]?t|did\s+not|doesn['’]?t|does\s+not)\s+(?:help|fix\s+(?:it|anything)|change\s+anything)|
    nothing\s+(?:changed|happen\w*)|
    (?:the\s+)?same\s+(?:
        error|issue|problem|failure|result|thing|output|exception|crash|bug|message|behaviou?r|story
    )s?|
    (?:the\s+)?(?:test|tests|build|ci|check|checks|lint|linter|app|page|server)\s+(?:is\s+|are\s+)?still|
    (?:try|run|do|test)\s+(?:it\s+|that\s+|this\s+)?again|
    retry\s+(?:it|that|this|with|using)|
    you\s+(?:
        didn['’]?t|did\s+not|forgot|missed|broke|still|haven['’]?t|have\s+not|skipped|ignored|removed|
        deleted|changed|reverted|introduced
    )|
    why\s+(?:did|didn['’]?t|would|are)\s+you|
    why\s+(?:is|does|did)\s+(?:it|that|this)|
    (?:undo|revert)\s+(?:it|that|this|those|these|the\s+last|your)|
    now\s+(?:
        it|that|this|they|i\s+get|i\s+see|there['’]?s|we\s+get|
        the\s+(?:test|tests|build|error|app|page|output)
    )|
    wrong\s+(?:
        file|one|place|function|method|directory|folder|approach|fix|answer|command|branch|output|result
    )|
    (?:i\s+)?(?:still\s+)?(?:get|got|getting|see|seeing)\s+(?:the\s+same|a\s+(?:new|different)|another)|
    go\s+ahead|continue\s+(?:with|where|from|on)|proceed\s+(?:with|to)|keep\s+going|carry\s+on|
    do\s+(?:it|that)|fix\s+(?:it|that|them)
";

static FOLLOW_UP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?ix)^{SEPARATORS}(?:(?:{ACKNOWLEDGEMENTS})\b{SEPARATORS})*(?:$|(?:{BACK_REFERENCES})\b|(?:no|nope)\s*[,.;:!])"
    ))
    .expect("follow-up pattern is valid")
});

static BARE_FOLLOW_UP: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(&format!(
        r"(?ix)^{SEPARATORS}(?:(?:{ACKNOWLEDGEMENTS}|{BACK_REFERENCES})\b{SEPARATORS})*$"
    ))
    .expect("bare follow-up pattern is valid")
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
                && Self::opens_task(summary, starts.is_empty())
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

    fn opens_task(summary: &str, first: bool) -> bool {
        if first {
            !Self::is_bare_follow_up(summary)
        } else {
            !Self::is_follow_up(summary)
        }
    }

    fn is_follow_up(summary: &str) -> bool {
        FOLLOW_UP.is_match(summary)
    }

    fn is_bare_follow_up(summary: &str) -> bool {
        BARE_FOLLOW_UP.is_match(summary)
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
    use jiff::Timestamp;
    use trodden_core::{HarnessId, SessionId};

    use super::*;

    const FOLLOW_UPS: &[&str] = &[
        "yes",
        "Yes!",
        "yep",
        "y",
        "ok",
        "OK.",
        "okay",
        "ok do it",
        "Okay, go ahead.",
        "yes please",
        "please do",
        "sure, go for it",
        "lgtm",
        "sounds good, continue",
        "continue",
        "Continue.",
        "please continue",
        "keep going",
        "proceed",
        "carry on",
        "go ahead",
        "go ahead and apply the fix",
        "Go ahead with the plan",
        "continue with the next step",
        "do it",
        "do it again but with logging enabled",
        "let's do it",
        "try again",
        "Try again with the --release flag",
        "retry",
        "retry with RUST_BACKTRACE=1",
        "one more time",
        "again",
        "still",
        "still failing",
        "Still failing on CI",
        "still broken",
        "still not working",
        "still doesn't work",
        "still getting the same error",
        "still the same",
        "same error",
        "Same error: TypeError: cannot read properties of undefined",
        "same issue on Windows",
        "the same problem happens in Safari",
        "same",
        "that didn't work",
        "That didn't work, the test still fails",
        "that did not help",
        "this broke the build",
        "that made it worse",
        "that's wrong",
        "That's not what I meant",
        "this is still failing",
        "it didn't work",
        "it still fails",
        "it's still broken",
        "its failing now",
        "It doesn’t work",
        "it keeps crashing on startup",
        "it says permission denied",
        "it throws a NullPointerException in OrderService",
        "it returns 500 for empty carts",
        "not working",
        "doesn't work",
        "didn't help",
        "nothing changed",
        "nothing happens when I click save",
        "the tests are still red",
        "the build is still failing",
        "CI still fails on the lint step",
        "now it fails with a different error",
        "Now I get a 404 on /api/orders",
        "now the page is blank",
        "I still get the same error",
        "getting a different error now",
        "I see another panic in the logs",
        "you forgot to update the tests",
        "you didn't change the config file",
        "You broke the login page",
        "why did you delete that file?",
        "why is it still failing",
        "why?",
        "undo that",
        "revert your last change",
        "revert it",
        "fix it",
        "fix that too",
        "fix it so the parser also handles quoted fields",
        "wrong file",
        "wrong",
        "nope",
        "no",
        "No.",
        "no, I meant the staging config",
        "No, use the other branch",
        "no! revert that",
        "hmm, still failing",
        "ok but it still fails",
        "that works, thanks",
        "thanks!",
        "👍",
        "...",
        "",
    ];

    const NEW_TASKS: &[&str] = &[
        "add dark mode",
        "Add dark mode",
        "fix the tests",
        "bump the version",
        "update dependencies",
        "run the linter",
        "format the code",
        "write a README",
        "Okay, next: implement CSV export",
        "OK now add a total subcommand",
        "ok, let's add pagination to the orders page",
        "Yes, and then add a migration for the new column",
        "No-op the logger in tests",
        "No tests cover the parser, add some",
        "Nowhere in the docs is the API described",
        "Notifications should batch per user",
        "Now add a total subcommand as well, printing the sum",
        "Still need to add tests for the CSV parser",
        "Continuous deployment is broken on main, fix the workflow",
        "Retry failed uploads with exponential backoff",
        "Revert the migration added in v2.3",
        "Undo support for the drawing canvas",
        "Do this: add a CSV export button to the reports page",
        "Fix this: the login redirect loops forever",
        "Wrong currency is shown on invoices for EU customers",
        "Why does the build take 10 minutes? Speed it up",
        "Same-day delivery option at checkout",
        "Sure-footed migration plan for the users table",
        "Okayish defaults for the retry policy",
        "Yesterday's deploy broke the cron jobs",
        "Go through the handlers and add request logging",
        "Proceeds from refunds are not subtracted from revenue",
        "Thanks to the new API we can drop the cache, remove it",
        "Again we need a feature flag for the beta",
        "Add a `count` subcommand that prints the number of invoices.",
        "Page 2 in src/paginate.js repeats the last product from page 1",
        "The cart total in src/cart.js ignores the discount code",
        "Explain how pagination works here",
        "k8s manifests need resource limits",
    ];

    const BARE: &[&str] = &[
        "continue",
        "yes",
        "ok do it",
        "try again",
        "still failing",
        "that didn't work",
        "it still doesn't work",
        "same error",
        "same error again",
        "go ahead",
        "nope, still broken",
        "fix it",
        "Okay, continue please.",
        "it's still not working for me either",
    ];

    const DESCRIPTIVE_FOLLOW_UPS: &[&str] = &[
        "it crashes when I upload a PNG larger than 5MB",
        "It returns 500 for empty carts",
        "That didn't work, the test still fails",
        "Same error: TypeError: cannot read properties of undefined",
        "go ahead and apply the fix",
        "no, I meant the staging config",
    ];

    fn trace(prompts: &[&str]) -> Trace {
        let events = prompts
            .iter()
            .enumerate()
            .map(|(seq, prompt)| Event {
                seq: u32::try_from(seq).expect("traces are small"),
                at: Timestamp::UNIX_EPOCH,
                kind: EventKind::Prompt {
                    summary: (*prompt).to_owned(),
                },
            })
            .collect();
        Trace {
            session: SessionId::new("0b6f7c1e-2d4a-4f0e-9a51-3c8e2f1d7b90"),
            harness: HarnessId::new("claude-code"),
            model: None,
            cwd: "/work/catalog".to_owned(),
            commit: None,
            started_at: Timestamp::UNIX_EPOCH,
            events,
        }
    }

    fn split<'a>(prompts: &[&'a str]) -> Vec<(&'a str, u32, u32)> {
        let trace = trace(prompts);
        Task::split(&trace)
            .iter()
            .map(|task| {
                let summary = prompts
                    .iter()
                    .copied()
                    .find(|prompt| *prompt == task.summary)
                    .expect("summaries come from prompts");
                (summary, task.first_seq(), task.last_seq())
            })
            .collect()
    }

    #[test]
    fn recognizes_follow_ups() {
        for prompt in FOLLOW_UPS {
            assert!(Task::is_follow_up(prompt), "{prompt:?} is a follow-up");
        }
    }

    #[test]
    fn recognizes_new_tasks() {
        for prompt in NEW_TASKS {
            assert!(!Task::is_follow_up(prompt), "{prompt:?} is a new task");
        }
    }

    #[test]
    fn bare_follow_ups_say_nothing_about_the_task() {
        for prompt in BARE {
            assert!(Task::is_bare_follow_up(prompt), "{prompt:?} is bare");
        }
        for prompt in DESCRIPTIVE_FOLLOW_UPS.iter().chain(NEW_TASKS) {
            assert!(!Task::is_bare_follow_up(prompt), "{prompt:?} is not bare");
        }
    }

    #[test]
    fn unrelated_short_and_acknowledged_prompts_start_their_own_tasks() {
        assert_eq!(
            split(&[
                "Page 2 in src/paginate.js repeats the last product from page 1",
                "add dark mode",
                "Okay, next: implement CSV export",
                "No-op the logger in tests",
            ]),
            [
                (
                    "Page 2 in src/paginate.js repeats the last product from page 1",
                    0,
                    0
                ),
                ("add dark mode", 1, 1),
                ("Okay, next: implement CSV export", 2, 2),
                ("No-op the logger in tests", 3, 3),
            ]
        );
    }

    #[test]
    fn follow_ups_join_the_current_task() {
        assert_eq!(
            split(&[
                "Page 2 in src/paginate.js repeats the last product from page 1",
                "still failing",
                "ok do it",
                "add dark mode",
                "that didn't work, the toggle does nothing",
                "yes",
            ]),
            [
                (
                    "Page 2 in src/paginate.js repeats the last product from page 1",
                    0,
                    2
                ),
                ("add dark mode", 3, 5),
            ]
        );
    }

    #[test]
    fn bare_follow_ups_never_name_a_task() {
        assert_eq!(split(&["continue", "yes", "still failing"]), []);
        assert_eq!(
            split(&["continue", "same error", "add dark mode", "try again"]),
            [("add dark mode", 2, 3)]
        );
        assert_eq!(
            split(&[
                "it crashes when I upload a PNG larger than 5MB",
                "still failing"
            ]),
            [("it crashes when I upload a PNG larger than 5MB", 0, 1)]
        );
    }
}
