use rand::Rng;
use rand_distr::{Beta, Distribution};
use trodden_core::procedure::Outcomes;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Evidence {
    pub successes: u32,
    pub failures: u32,
}

impl Evidence {
    pub const fn new(successes: u32, failures: u32) -> Self {
        Self {
            successes,
            failures,
        }
    }

    pub const fn of(outcomes: &Outcomes) -> Self {
        Self::new(outcomes.successes, outcomes.failures)
    }

    pub const fn total(self) -> u32 {
        self.successes.saturating_add(self.failures)
    }

    #[must_use]
    pub const fn record(self, succeeded: bool) -> Self {
        if succeeded {
            Self::new(self.successes.saturating_add(1), self.failures)
        } else {
            Self::new(self.successes, self.failures.saturating_add(1))
        }
    }

    pub fn mean(self) -> f64 {
        (f64::from(self.successes) + 1.0) / (f64::from(self.total()) + 2.0)
    }

    pub fn sample<R: Rng + ?Sized>(self, rng: &mut R) -> f64 {
        Beta::new(self.alpha(), self.beta())
            .expect("Beta parameters are at least 1")
            .sample(rng)
    }

    pub fn prob_better_than(self, other: Self) -> f64 {
        if self.successes > other.successes {
            return 1.0 - other.prob_better_than(self);
        }
        let (a, b) = (self.successes + 1, self.failures + 1);
        let (c, d) = (other.successes + 1, other.failures + 1);
        let ln_gamma = LnGamma::up_to(a + b + c + d);
        let ln_beta = |x: u32, y: u32| ln_gamma.of(x) + ln_gamma.of(y) - ln_gamma.of(x + y);
        let total: f64 = (0..a)
            .map(|i| {
                (ln_beta(c + i, b + d) - f64::from(b + i).ln() - ln_beta(1 + i, b) - ln_beta(c, d))
                    .exp()
            })
            .sum();
        total.clamp(0.0, 1.0)
    }

    pub fn prob_below(self, rate: f64) -> f64 {
        if rate <= 0.0 {
            return 0.0;
        }
        if rate >= 1.0 {
            return 1.0;
        }
        let a = self.successes + 1;
        let n = self.total() + 1;
        let ln_gamma = LnGamma::up_to(n + 2);
        let ln_choose = |k: u32| ln_gamma.of(n + 1) - ln_gamma.of(k + 1) - ln_gamma.of(n - k + 1);
        let total: f64 = (a..=n)
            .map(|k| {
                (ln_choose(k) + f64::from(k) * rate.ln() + f64::from(n - k) * (1.0 - rate).ln())
                    .exp()
            })
            .sum();
        total.clamp(0.0, 1.0)
    }

    fn alpha(self) -> f64 {
        f64::from(self.successes) + 1.0
    }

    fn beta(self) -> f64 {
        f64::from(self.failures) + 1.0
    }
}

struct LnGamma(Vec<f64>);

impl LnGamma {
    fn up_to(max: u32) -> Self {
        let mut table = Vec::with_capacity(max as usize + 1);
        table.push(0.0);
        let mut sum = 0.0;
        for n in 1..=max {
            table.push(sum);
            sum += f64::from(n).ln();
        }
        Self(table)
    }

    fn of(&self, n: u32) -> f64 {
        self.0[n as usize]
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use rand::{SeedableRng, rngs::SmallRng};

    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn known_comparisons() {
        assert!(close(
            Evidence::new(1, 0).prob_better_than(Evidence::default()),
            2.0 / 3.0,
            1e-12
        ));
        assert!(close(
            Evidence::new(4, 6).prob_better_than(Evidence::new(4, 6)),
            0.5,
            1e-9
        ));
        assert!(Evidence::new(9, 1).prob_better_than(Evidence::new(2, 8)) > 0.99);
    }

    #[test]
    fn comparison_matches_sampling() {
        let mut rng = SmallRng::seed_from_u64(7);
        let (x, y) = (Evidence::new(7, 3), Evidence::new(5, 4));
        let draws = 200_000;
        let wins = (0..draws)
            .filter(|_| x.sample(&mut rng) > y.sample(&mut rng))
            .count();
        let sampled = wins as f64 / f64::from(draws);

        assert!(
            close(x.prob_better_than(y), sampled, 0.005),
            "{sampled} vs {}",
            x.prob_better_than(y)
        );
    }

    #[test]
    fn tail_probabilities() {
        assert!(close(Evidence::default().prob_below(0.3), 0.3, 1e-12));
        assert!(close(Evidence::new(0, 1).prob_below(0.5), 0.75, 1e-12));
        assert!(Evidence::new(0, 9).prob_below(0.5) > 0.99);
    }

    quickcheck! {
        fn comparisons_are_complementary(a: u8, b: u8, c: u8, d: u8) -> bool {
            let x = Evidence::new(a.into(), b.into());
            let y = Evidence::new(c.into(), d.into());
            close(x.prob_better_than(y) + y.prob_better_than(x), 1.0, 1e-6)
        }

        fn more_successes_never_hurt(a: u8, b: u8, c: u8, d: u8) -> bool {
            let x = Evidence::new(a.into(), b.into());
            let y = Evidence::new(c.into(), d.into());
            x.record(true).prob_better_than(y) >= x.prob_better_than(y) - 1e-9
        }
    }
}
