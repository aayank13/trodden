use super::{Procedure, Slot, SlotKind};

const MAX_DIFFERING_WORDS: usize = 2;

impl Procedure {
    #[must_use]
    pub fn anti_unify(&self, other: &Self) -> Option<Self> {
        if self.steps.len() != other.steps.len() {
            return None;
        }
        let mut merged = self.clone();
        let mut slots = Slots {
            mine: &self.slots,
            theirs: &other.slots,
            merged: self.slots.clone(),
            pairs: Vec::new(),
        };
        for (step, theirs) in merged.steps.iter_mut().zip(&other.steps) {
            if step.kind != theirs.kind || step.target != theirs.target {
                return None;
            }
            match (&step.command, &theirs.command) {
                (None, None) => {}
                (Some(mine), Some(theirs)) => step.command = Some(slots.unify(mine, theirs)?),
                _ => return None,
            }
        }
        if let (Some(verify), Some(theirs)) = (&mut merged.verify, &other.verify) {
            let mut attempt = slots.clone();
            if let Some(command) = attempt.unify(&verify.command, &theirs.command) {
                verify.command = command;
                slots = attempt;
            }
        }
        merged.slots = slots.merged;
        Some(merged)
    }
}

#[derive(Debug, Clone)]
struct Slots<'a> {
    mine: &'a [Slot],
    theirs: &'a [Slot],
    merged: Vec<Slot>,
    pairs: Vec<([String; 2], String)>,
}

impl Slots<'_> {
    fn unify(&mut self, mine: &str, theirs: &str) -> Option<String> {
        if mine == theirs {
            return Some(mine.to_owned());
        }
        let my_words: Vec<&str> = mine.split_whitespace().collect();
        let their_words: Vec<&str> = theirs.split_whitespace().collect();
        if my_words.len() != their_words.len() || my_words.first() != their_words.first() {
            return None;
        }
        let differing = my_words
            .iter()
            .zip(&their_words)
            .filter(|(mine, theirs)| mine != theirs)
            .count();
        if differing > MAX_DIFFERING_WORDS {
            return None;
        }
        let mut words = Vec::with_capacity(my_words.len());
        for (index, (mine, theirs)) in my_words.iter().zip(&their_words).enumerate() {
            if mine == theirs {
                words.push((*mine).to_owned());
            } else {
                let before = [&my_words[..index], &their_words[..index]];
                words.push(self.word(before, mine, theirs)?);
            }
        }
        Some(words.join(" "))
    }

    fn word(&mut self, before: [&[&str]; 2], mine: &str, theirs: &str) -> Option<String> {
        let (prefix, mine_value, their_value) = match (mine.split_once('='), theirs.split_once('='))
        {
            (Some((key, mine_value)), Some((their_key, their_value))) if key == their_key => {
                (format!("{key}="), mine_value, their_value)
            }
            _ => (String::new(), mine, theirs),
        };
        if prefix.is_empty() && (mine.starts_with('-') || theirs.starts_with('-')) {
            return None;
        }
        let base = Self::base_name(before, &prefix);
        let word = match (mine_value.contains('{'), their_value.contains('{')) {
            (false, false) => self.literals(&base, mine_value, their_value),
            (true, false) => {
                let fit = Fit::find(self.mine, mine_value, their_value)?;
                let pair = [fit.placeholder(), fit.value.to_owned()];
                fit.render(&self.bind(&base, pair, Some(fit.slot), &[fit.value]))
            }
            (false, true) => {
                let fit = Fit::find(self.theirs, their_value, mine_value)?;
                let pair = [fit.value.to_owned(), fit.placeholder()];
                fit.render(&self.bind(&base, pair, Some(fit.slot), &[fit.value]))
            }
            (true, true) => return None,
        };
        Some(format!("{prefix}{word}"))
    }

    fn literals(&mut self, base: &str, mine: &str, theirs: &str) -> String {
        for slot in self.mine {
            let Some(their_slot) = self.theirs.iter().find(|theirs| theirs.name == slot.name)
            else {
                continue;
            };
            let placeholder = format!("{{{}}}", slot.name);
            for my_example in &slot.examples {
                for their_example in &their_slot.examples {
                    let generalized = mine.replacen(my_example.as_str(), &placeholder, 1);
                    if generalized.contains(&placeholder)
                        && generalized == theirs.replacen(their_example.as_str(), &placeholder, 1)
                    {
                        return generalized;
                    }
                }
            }
        }
        let pair = [mine.to_owned(), theirs.to_owned()];
        format!("{{{}}}", self.bind(base, pair, None, &[mine, theirs]))
    }

    fn bind(
        &mut self,
        base: &str,
        pair: [String; 2],
        origin: Option<&Slot>,
        values: &[&str],
    ) -> String {
        if let Some((_, name)) = self.pairs.iter().find(|(known, _)| *known == pair) {
            return name.clone();
        }
        let name = match origin {
            Some(slot) if !self.pairs.iter().any(|(_, name)| *name == slot.name) => {
                slot.name.clone()
            }
            _ => self.fresh_name(base),
        };
        if !self.merged.iter().any(|slot| slot.name == name) {
            self.merged.push(Slot {
                name: name.clone(),
                kind: origin.map_or(SlotKind::Number, |slot| slot.kind),
                examples: Vec::new(),
            });
        }
        let known = origin.into_iter().flat_map(|slot| &slot.examples);
        for value in known.map(String::as_str).chain(values.iter().copied()) {
            self.record(&name, value);
        }
        self.pairs.push((pair, name.clone()));
        name
    }

    fn record(&mut self, name: &str, value: &str) {
        if let Some(slot) = self.merged.iter_mut().find(|slot| slot.name == name) {
            slot.kind = Self::widen(slot.kind, value);
            if !slot.examples.iter().any(|known| known == value)
                && slot.examples.len() < Slot::MAX_EXAMPLES
            {
                slot.examples.push(value.to_owned());
            }
        }
    }

    fn fresh_name(&self, base: &str) -> String {
        let taken = |name: &str| {
            self.merged
                .iter()
                .chain(self.theirs)
                .any(|slot| slot.name == name)
        };
        let mut name = base.to_owned();
        let mut suffix = 2;
        while taken(&name) {
            name = format!("{base}_{suffix}");
            suffix += 1;
        }
        name
    }

    fn base_name(before: [&[&str]; 2], prefix: &str) -> String {
        const TEST_RUNNERS: &[&str] = &["test", "pytest", "jest", "vitest", "rspec", "mocha"];
        let option = if prefix.is_empty() {
            before[0]
                .last()
                .filter(|word| {
                    word.starts_with('-')
                        && !word.trim_start_matches('-').is_empty()
                        && !word.contains('=')
                })
                .copied()
        } else {
            Some(prefix.trim_end_matches('='))
        };
        match option {
            Some(option) => option
                .trim_start_matches('-')
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() {
                        c.to_ascii_lowercase()
                    } else {
                        '_'
                    }
                })
                .collect(),
            None if before.iter().flat_map(|words| words.iter()).any(|word| {
                let program = word.rsplit('/').next().unwrap_or(word);
                TEST_RUNNERS.contains(&program)
            }) =>
            {
                "test".to_owned()
            }
            None => "arg".to_owned(),
        }
    }

    fn widen(kind: SlotKind, value: &str) -> SlotKind {
        match kind {
            SlotKind::Number | SlotKind::Identifier if value.contains('/') => SlotKind::Path,
            SlotKind::Number if value.parse::<f64>().is_err() => SlotKind::Identifier,
            kind => kind,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Fit<'s> {
    slot: &'s Slot,
    before: &'s str,
    after: &'s str,
    value: &'s str,
}

impl<'s> Fit<'s> {
    fn find(slots: &'s [Slot], template: &'s str, literal: &'s str) -> Option<Self> {
        slots.iter().find_map(|slot| {
            let (before, after) = template.split_once(&format!("{{{}}}", slot.name))?;
            let value = literal.strip_prefix(before)?.strip_suffix(after)?;
            (!value.is_empty() && !after.contains('{')).then_some(Self {
                slot,
                before,
                after,
                value,
            })
        })
    }

    fn placeholder(&self) -> String {
        format!("{{{}}}", self.slot.name)
    }

    fn render(&self, name: &str) -> String {
        format!("{}{{{name}}}{}", self.before, self.after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedure::{Step, StepKind, Verification};

    fn procedure(commands: &[&str]) -> Procedure {
        let mut procedure: Procedure = Procedure::example();
        procedure.steps = commands
            .iter()
            .map(|command| Step {
                kind: StepKind::Run,
                command: Some((*command).to_owned()),
                target: None,
                symbols: Vec::new(),
                reads: Vec::new(),
                writes: Vec::new(),
            })
            .collect();
        procedure.slots = Vec::new();
        procedure.verify = None;
        procedure
    }

    fn commands(procedure: &Procedure) -> Vec<&str> {
        procedure
            .steps
            .iter()
            .filter_map(|step| step.command.as_deref())
            .collect()
    }

    fn checked(commands: &[&str], check: &str) -> Procedure {
        let mut procedure = procedure(commands);
        procedure.verify = Some(Verification {
            command: check.to_owned(),
            expect_exit: 0,
            declared_by: None,
        });
        procedure
    }

    fn check(procedure: &Procedure) -> Option<&str> {
        procedure
            .verify
            .as_ref()
            .map(|verify| verify.command.as_str())
    }

    #[derive(Debug, PartialEq)]
    struct Shape {
        lines: Vec<String>,
        slots: Vec<(String, SlotKind, Vec<String>)>,
    }

    impl Shape {
        fn of(procedure: &Procedure) -> Self {
            let mut lines: Vec<String> =
                commands(procedure).into_iter().map(str::to_owned).collect();
            lines.extend(check(procedure).map(str::to_owned));
            let mut slots: Vec<(String, SlotKind, Vec<String>)> = Vec::new();
            for line in &lines {
                for piece in line.split('{').skip(1) {
                    let (name, _) = piece.split_once('}').expect("placeholders are closed");
                    let slot = procedure
                        .slots
                        .iter()
                        .find(|slot| slot.name == name)
                        .expect("placeholders name a slot");
                    if !slots.iter().any(|(known, _, _)| known == name) {
                        let mut examples = slot.examples.clone();
                        examples.sort();
                        slots.push((slot.name.clone(), slot.kind, examples));
                    }
                }
            }
            assert_eq!(slots.len(), procedure.slots.len(), "every slot is used");
            Self { lines, slots }
        }

        fn renamed(mut self) -> Self {
            for (index, (name, _, _)) in self.slots.iter_mut().enumerate() {
                for line in &mut self.lines {
                    *line = line.replace(&format!("{{{name}}}"), &format!("{{{index}}}"));
                }
                *name = index.to_string();
            }
            self
        }
    }

    fn session(test: &str, dir: &str, release: bool) -> Procedure {
        let check = if release {
            format!("cargo test --release {test}")
        } else {
            format!("cargo test {test}")
        };
        checked(
            &[
                &format!("mkdir {dir}"),
                &format!("python3 tools/new.py --name={test} {dir}"),
                &check,
            ],
            &check,
        )
    }

    #[test]
    fn differing_arguments_become_slots() {
        let first = procedure(&[
            "python3 tools/new.py --name add_priority",
            "pytest -k priority",
        ]);
        let second = procedure(&["python3 tools/new.py --name add_due_date", "pytest -k due"]);

        let merged = first.anti_unify(&second).expect("same path");

        assert_eq!(
            commands(&merged),
            ["python3 tools/new.py --name {name}", "pytest -k {k}"]
        );
        assert_eq!(merged.slots[0].examples, ["add_priority", "add_due_date"]);
    }

    #[test]
    fn a_third_session_adds_a_value() {
        let first = procedure(&["cargo test count_prints"]);
        let second = procedure(&["cargo test total_prints"]);
        let third = procedure(&["cargo test unpaid_prints"]);

        let merged = first
            .anti_unify(&second)
            .and_then(|merged| merged.anti_unify(&third))
            .expect("same path");

        assert_eq!(commands(&merged), ["cargo test {test}"]);
        assert_eq!(merged.slots.len(), 1);
        assert_eq!(
            merged.slots[0].examples,
            ["count_prints", "total_prints", "unpaid_prints"]
        );
    }

    #[test]
    fn values_of_shared_slots_are_reused() {
        let mut first = procedure(&["python3 tools/apply.py migrations/0005_add_priority.sql"]);
        first.slots = vec![Slot {
            name: "migration".to_owned(),
            kind: SlotKind::Identifier,
            examples: vec!["0005_add_priority".to_owned()],
        }];
        let mut second = procedure(&["python3 tools/apply.py migrations/0006_add_due.sql"]);
        second.slots = vec![Slot {
            name: "migration".to_owned(),
            kind: SlotKind::Identifier,
            examples: vec!["0006_add_due".to_owned()],
        }];

        let merged = first.anti_unify(&second).expect("same path");

        assert_eq!(
            commands(&merged),
            ["python3 tools/apply.py migrations/{migration}.sql"]
        );
        assert_eq!(merged.slots.len(), 1);
    }

    #[test]
    fn different_paths_stay_apart() {
        let base = procedure(&["cargo test count_prints"]);
        for other in [
            procedure(&["cargo test --release count_prints"]),
            procedure(&["cargo test --doc"]),
            procedure(&["cargo nextest run count_prints"]),
            procedure(&["cargo build", "cargo test count_prints"]),
            procedure(&["cargo test count_prints extra args here"]),
        ] {
            assert!(base.anti_unify(&other).is_none(), "{:?}", commands(&other));
        }
        let wide = procedure(&["make a b c"]);
        assert!(wide.anti_unify(&procedure(&["make x y z"])).is_none());
    }

    #[test]
    fn the_check_shares_the_slot_of_its_step() {
        let first = checked(&["cargo test count_prints"], "cargo test count_prints");
        let second = checked(&["cargo test total_prints"], "cargo test total_prints");

        let merged = first.anti_unify(&second).expect("same path");

        assert_eq!(commands(&merged), ["cargo test {test}"]);
        assert_eq!(check(&merged), Some("cargo test {test}"));
        assert_eq!(merged.slots.len(), 1);
        assert_eq!(merged.slots[0].examples, ["count_prints", "total_prints"]);
    }

    #[test]
    fn values_that_stop_agreeing_get_their_own_slot() {
        let merged = checked(&["cargo test count_prints"], "cargo test count_prints")
            .anti_unify(&checked(
                &["cargo test total_prints"],
                "cargo test total_prints",
            ))
            .and_then(|merged| {
                merged.anti_unify(&checked(
                    &["cargo test unpaid_prints"],
                    "cargo test paid_prints",
                ))
            })
            .expect("same path");

        assert_eq!(commands(&merged), ["cargo test {test}"]);
        assert_eq!(check(&merged), Some("cargo test {test_2}"));
        assert_eq!(
            merged.slots[0].examples,
            ["count_prints", "total_prints", "unpaid_prints"]
        );
        assert_eq!(
            merged.slots[1].examples,
            ["count_prints", "total_prints", "paid_prints"]
        );
    }

    #[test]
    fn placeholders_on_either_side_stay_placeholders() {
        let merged = checked(&["cargo test count_prints"], "cargo test count_prints")
            .anti_unify(&checked(
                &["cargo test total_prints"],
                "cargo test total_prints",
            ))
            .expect("same path");
        let third = checked(&["cargo test unpaid_prints"], "cargo test unpaid_prints");

        let forward = merged.anti_unify(&third).expect("same path");
        let backward = third.anti_unify(&merged).expect("same path");

        assert_eq!(commands(&backward), ["cargo test {test}"]);
        assert_eq!(check(&backward), Some("cargo test {test}"));
        assert_eq!(
            backward.slots[0].examples,
            ["count_prints", "total_prints", "unpaid_prints"]
        );
        assert_eq!(backward.slots, forward.slots);
    }

    #[test]
    fn braces_that_are_not_placeholders_are_not_values() {
        let named = procedure(&["find src -exec rustfmt main.rs +"]);
        let braced = procedure(&["find src -exec rustfmt {} +"]);

        assert!(named.anti_unify(&braced).is_none());
        assert!(braced.anti_unify(&named).is_none());
    }

    #[test]
    fn merging_does_not_depend_on_order() {
        let mut sessions = Vec::new();
        for test in ["count_prints", "total_prints", "out"] {
            for dir in ["out", "build/out", "total_prints"] {
                for release in [false, true] {
                    sessions.push(session(test, dir, release));
                }
            }
        }
        let mut outcomes = [0_usize; 2];
        for a in &sessions {
            for b in &sessions {
                let forward = a.anti_unify(b).as_ref().map(Shape::of);
                let backward = b.anti_unify(a).as_ref().map(Shape::of);
                assert_eq!(forward, backward, "{:?} and {:?}", commands(a), commands(b));
                for c in &sessions {
                    let merges: Vec<Option<Shape>> = [[a, b, c], [a, c, b], [b, c, a]]
                        .into_iter()
                        .flat_map(|[x, y, z]| {
                            [
                                x.anti_unify(y).and_then(|merged| merged.anti_unify(z)),
                                y.anti_unify(z).and_then(|merged| x.anti_unify(&merged)),
                            ]
                        })
                        .map(|merged| merged.as_ref().map(|merged| Shape::of(merged).renamed()))
                        .collect();
                    outcomes[usize::from(merges[0].is_some())] += 1;
                    assert!(
                        merges.windows(2).all(|pair| pair[0] == pair[1]),
                        "{:?}, {:?} and {:?}: {merges:#?}",
                        commands(a),
                        commands(b),
                        commands(c)
                    );
                }
            }
        }
        assert!(outcomes.iter().all(|&count| count > 0), "{outcomes:?}");
    }
}
