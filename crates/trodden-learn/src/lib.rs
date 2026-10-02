mod evidence;
mod policy;

use jiff::{SignedDuration, Timestamp};
use trodden_core::procedure::Lifecycle;

pub use evidence::Evidence;
pub use policy::{Policy, RevisionEvidence};

#[derive(Debug)]
pub struct Holdout;

impl Holdout {
    pub const INITIAL_RATE: f64 = 0.10;
    pub const ESTABLISHED_RATE: f64 = 0.02;
    pub const MIN_OUTCOMES: u32 = 10;
    pub const ENOUGH_OUTCOMES: u32 = 50;

    pub fn rate(configured: Option<f64>, injected: Evidence, held_out: Evidence) -> f64 {
        if let Some(rate) = configured {
            return rate.clamp(0.0, 1.0);
        }
        if Self::is_established(injected, held_out) {
            Self::ESTABLISHED_RATE
        } else {
            Self::INITIAL_RATE
        }
    }

    pub fn is_established(injected: Evidence, held_out: Evidence) -> bool {
        if held_out.total() >= Self::ENOUGH_OUTCOMES {
            return true;
        }
        if held_out.total() < Self::MIN_OUTCOMES || injected.total() < Self::MIN_OUTCOMES {
            return false;
        }
        let better = injected.prob_better_than(held_out);
        better >= Policy::CONFIDENCE || better <= 1.0 - Policy::CONFIDENCE
    }
}

#[derive(Debug)]
pub struct Aging;

impl Aging {
    pub const STALE_AFTER: SignedDuration = SignedDuration::from_hours(30 * 24);
    pub const ARCHIVE_AFTER: SignedDuration = SignedDuration::from_hours(90 * 24);

    pub fn state(state: Lifecycle, last_used: Timestamp, now: Timestamp) -> Lifecycle {
        let idle = now.duration_since(last_used);
        match state {
            Lifecycle::Active | Lifecycle::Stale | Lifecycle::Candidate
                if idle >= Self::ARCHIVE_AFTER =>
            {
                Lifecycle::Archived
            }
            Lifecycle::Active if idle >= Self::STALE_AFTER => Lifecycle::Stale,
            Lifecycle::Stale if idle < Self::STALE_AFTER => Lifecycle::Active,
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdouts_shrink_once_the_effect_is_known() {
        let unknown = Holdout::rate(None, Evidence::new(3, 1), Evidence::new(1, 1));
        let clear = Holdout::rate(None, Evidence::new(18, 2), Evidence::new(4, 8));
        let configured = Holdout::rate(Some(0.0), Evidence::new(3, 1), Evidence::new(1, 1));

        assert!((unknown - Holdout::INITIAL_RATE).abs() < f64::EPSILON);
        assert!((clear - Holdout::ESTABLISHED_RATE).abs() < f64::EPSILON);
        assert!(configured.abs() < f64::EPSILON);
    }

    #[test]
    fn unused_revisions_age() {
        let now: Timestamp = "2026-12-31T00:00:00Z".parse().expect("valid timestamp");
        let days_ago = |days: i64| now - SignedDuration::from_hours(days * 24);

        assert_eq!(
            Aging::state(Lifecycle::Active, days_ago(3), now),
            Lifecycle::Active
        );
        assert_eq!(
            Aging::state(Lifecycle::Active, days_ago(31), now),
            Lifecycle::Stale
        );
        assert_eq!(
            Aging::state(Lifecycle::Stale, days_ago(1), now),
            Lifecycle::Active
        );
        assert_eq!(
            Aging::state(Lifecycle::Candidate, days_ago(91), now),
            Lifecycle::Archived
        );
        assert_eq!(
            Aging::state(Lifecycle::Quarantined, days_ago(1), now),
            Lifecycle::Quarantined
        );
    }
}
