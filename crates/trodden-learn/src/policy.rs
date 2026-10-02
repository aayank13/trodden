use rand::Rng;
use trodden_core::procedure::Lifecycle;

use crate::Evidence;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RevisionEvidence {
    pub revision: u32,
    pub state: Lifecycle,
    pub evidence: Evidence,
}

#[derive(Debug)]
pub struct Policy;

impl Policy {
    pub const CONFIDENCE: f64 = 0.95;
    pub const PROMOTION_MARGIN: f64 = 0.05;
    pub const MIN_PROMOTION_OUTCOMES: u32 = 3;
    pub const MIN_QUARANTINE_OUTCOMES: u32 = 5;
    pub const MIN_HOLDOUT_OUTCOMES: u32 = 3;

    pub fn choose<R: Rng + ?Sized>(evidence: &[Evidence], rng: &mut R) -> Option<usize> {
        evidence
            .iter()
            .map(|evidence| evidence.sample(rng))
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index)
    }

    pub fn settle(revisions: &[RevisionEvidence], holdout: Evidence) -> Vec<(u32, Lifecycle)> {
        let mut changes = Vec::new();
        let serving = |state| {
            matches!(
                state,
                Lifecycle::Active | Lifecycle::Stale | Lifecycle::Candidate
            )
        };

        let mut remaining: Vec<RevisionEvidence> = Vec::new();
        for revision in revisions.iter().filter(|r| serving(r.state)) {
            if Self::is_harmful(revision.evidence, holdout) {
                changes.push((revision.revision, Lifecycle::Quarantined));
            } else {
                remaining.push(*revision);
            }
        }

        let incumbent = remaining
            .iter()
            .find(|r| matches!(r.state, Lifecycle::Active | Lifecycle::Stale));
        let best_candidate = remaining
            .iter()
            .filter(|r| r.state == Lifecycle::Candidate)
            .max_by(|a, b| {
                a.evidence
                    .mean()
                    .total_cmp(&b.evidence.mean())
                    .then(a.revision.cmp(&b.revision))
            });
        match (incumbent, best_candidate) {
            (None, Some(candidate)) => changes.push((candidate.revision, Lifecycle::Active)),
            (Some(incumbent), Some(candidate))
                if Self::beats(candidate.evidence, incumbent.evidence) =>
            {
                changes.push((candidate.revision, Lifecycle::Active));
                changes.push((incumbent.revision, Lifecycle::Candidate));
            }
            _ => {}
        }
        changes
    }

    fn is_harmful(evidence: Evidence, holdout: Evidence) -> bool {
        if evidence.total() < Self::MIN_QUARANTINE_OUTCOMES {
            return false;
        }
        if holdout.total() >= Self::MIN_HOLDOUT_OUTCOMES {
            holdout.prob_better_than(evidence) >= Self::CONFIDENCE
        } else {
            evidence.prob_below(0.5) >= Self::CONFIDENCE
        }
    }

    fn beats(candidate: Evidence, incumbent: Evidence) -> bool {
        candidate.total() >= Self::MIN_PROMOTION_OUTCOMES
            && candidate.mean() >= incumbent.mean() + Self::PROMOTION_MARGIN
            && candidate.prob_better_than(incumbent) >= Self::CONFIDENCE
    }
}

#[cfg(test)]
mod tests {
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    fn revision(
        revision: u32,
        state: Lifecycle,
        successes: u32,
        failures: u32,
    ) -> RevisionEvidence {
        RevisionEvidence {
            revision,
            state,
            evidence: Evidence::new(successes, failures),
        }
    }

    #[test]
    fn promotes_a_clearly_better_candidate() {
        let changes = Policy::settle(
            &[
                revision(1, Lifecycle::Active, 2, 6),
                revision(2, Lifecycle::Candidate, 8, 0),
            ],
            Evidence::default(),
        );

        assert_eq!(changes, [(2, Lifecycle::Active), (1, Lifecycle::Candidate)]);
    }

    #[test]
    fn keeps_the_incumbent_on_thin_evidence() {
        let changes = Policy::settle(
            &[
                revision(1, Lifecycle::Active, 3, 1),
                revision(2, Lifecycle::Candidate, 2, 0),
            ],
            Evidence::default(),
        );

        assert!(changes.is_empty(), "{changes:?}");
    }

    #[test]
    fn quarantines_revisions_worse_than_the_holdout() {
        let changes = Policy::settle(
            &[
                revision(1, Lifecycle::Active, 1, 7),
                revision(2, Lifecycle::Candidate, 1, 0),
            ],
            Evidence::new(6, 1),
        );

        assert_eq!(
            changes,
            [(1, Lifecycle::Quarantined), (2, Lifecycle::Active)]
        );
    }

    #[test]
    fn without_a_holdout_only_clear_failure_quarantines() {
        let failing = Policy::settle(&[revision(1, Lifecycle::Active, 0, 6)], Evidence::default());
        let mediocre = Policy::settle(&[revision(1, Lifecycle::Active, 3, 4)], Evidence::default());

        assert_eq!(failing, [(1, Lifecycle::Quarantined)]);
        assert!(mediocre.is_empty());
    }

    #[test]
    fn thompson_sampling_favors_the_better_revision() {
        let mut rng = SmallRng::seed_from_u64(1);
        let evidence = [Evidence::new(2, 8), Evidence::new(8, 2)];

        let chosen = (0..1_000)
            .filter(|_| Policy::choose(&evidence, &mut rng) == Some(1))
            .count();

        assert!(chosen > 950, "{chosen}");
        assert_eq!(Policy::choose(&[], &mut rng), None);
    }
}
