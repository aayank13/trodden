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
            mine: merged.slots.clone(),
            theirs: &other.slots,
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
        merged.slots = slots.mine;
        Some(merged)
    }
}

#[derive(Debug, Clone)]
struct Slots<'a> {
    mine: Vec<Slot>,
    theirs: &'a [Slot],
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
                words.push(self.word(&my_words[..index], mine, theirs)?);
            }
        }
        Some(words.join(" "))
    }

    fn word(&mut self, before: &[&str], mine: &str, theirs: &str) -> Option<String> {
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

        if let Some((name, value)) = self.fit(mine_value, their_value) {
            self.record(&name, value);
            return Some(format!("{prefix}{mine_value}"));
        }
        if mine_value.contains('{') {
            return None;
        }

        for slot in &self.mine {
            let Some(their_slot) = self.theirs.iter().find(|theirs| theirs.name == slot.name)
            else {
                continue;
            };
            let placeholder = format!("{{{}}}", slot.name);
            for my_example in &slot.examples {
                for their_example in &their_slot.examples {
                    let generalized = mine_value.replacen(my_example.as_str(), &placeholder, 1);
                    if generalized.contains(&placeholder)
                        && generalized
                            == their_value.replacen(their_example.as_str(), &placeholder, 1)
                    {
                        return Some(format!("{prefix}{generalized}"));
                    }
                }
            }
        }

        let name = self.fresh_name(before, &prefix);
        self.mine.push(Slot {
            name: name.clone(),
            kind: Self::kind_of(mine_value),
            examples: Vec::new(),
        });
        self.record(&name, mine_value);
        self.record(&name, their_value);
        Some(format!("{prefix}{{{name}}}"))
    }

    fn fit<'w>(&self, template: &str, literal: &'w str) -> Option<(String, &'w str)> {
        self.mine.iter().find_map(|slot| {
            let placeholder = format!("{{{}}}", slot.name);
            let (before, after) = template.split_once(&placeholder)?;
            let value = literal.strip_prefix(before)?.strip_suffix(after)?;
            (!value.is_empty() && !after.contains('{')).then(|| (slot.name.clone(), value))
        })
    }

    fn record(&mut self, name: &str, value: &str) {
        if let Some(slot) = self.mine.iter_mut().find(|slot| slot.name == name)
            && !slot.examples.iter().any(|known| known == value)
            && slot.examples.len() < Slot::MAX_EXAMPLES
        {
            slot.examples.push(value.to_owned());
        }
    }

    fn fresh_name(&self, before: &[&str], prefix: &str) -> String {
        const TEST_RUNNERS: &[&str] = &["test", "pytest", "jest", "vitest", "rspec", "mocha"];
        let option = if prefix.is_empty() {
            before
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
        let base: String = match option {
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
            None if before.iter().any(|word| {
                let program = word.rsplit('/').next().unwrap_or(word);
                TEST_RUNNERS.contains(&program)
            }) =>
            {
                "test".to_owned()
            }
            None => "arg".to_owned(),
        };
        let mut name = base.clone();
        let mut suffix = 2;
        while self.mine.iter().any(|slot| slot.name == name) {
            name = format!("{base}_{suffix}");
            suffix += 1;
        }
        name
    }

    fn kind_of(value: &str) -> SlotKind {
        if value.contains('/') {
            SlotKind::Path
        } else if value.parse::<f64>().is_ok() {
            SlotKind::Number
        } else {
            SlotKind::Identifier
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::procedure::{Step, StepKind};

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
}
