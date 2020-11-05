use num_traits::Float;
use ordered_float::OrderedFloat;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Bernoulli, Uniform};
use rand::prelude::*;
use std::marker::PhantomData;
use std::time::Duration;

pub(crate) trait Mutate<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    fn mutate<R>(&self, xs: &mut [T], rng: &mut R)
    where
        R: Rng + ?Sized;

    fn copy_mutate<R>(&self, xs: &[T], rng: &mut R) -> Vec<T>
    where
        R: Rng + ?Sized,
    {
        let mut xs = xs.to_vec();
        self.mutate(&mut xs, rng);
        xs
    }
}

#[derive(Debug, Clone)]
pub(crate) struct VecMutation<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    tweak_dist: Bernoulli,
    range_dist: D,
    _t: PhantomData<T>,
}

impl<T, D> VecMutation<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    #[allow(dead_code)]
    fn new(tweak_dist: Bernoulli, range_dist: D) -> VecMutation<T, D> {
        VecMutation {
            tweak_dist,
            range_dist,
            _t: PhantomData,
        }
    }
}

impl<T, D> Mutate<T, D> for VecMutation<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    fn mutate<R>(&self, xs: &mut [T], rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        for x in xs {
            if self.tweak_dist.sample(rng) {
                *x = self.range_dist.sample(rng);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PointMutation<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    range_dist: D,
    point_dist: Uniform<usize>,
    _t: PhantomData<T>,
}

impl<T, D> PointMutation<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    #[allow(dead_code)]
    fn new(range_dist: D, n: usize) -> PointMutation<T, D> {
        PointMutation {
            range_dist,
            point_dist: Uniform::new(0, n),
            _t: PhantomData,
        }
    }
}

impl<T, D> Mutate<T, D> for PointMutation<T, D>
where
    T: SampleUniform + Clone,
    D: Distribution<T>,
{
    fn mutate<R>(&self, xs: &mut [T], rng: &mut R)
    where
        R: Rng + ?Sized,
    {
        let point = self.point_dist.sample(rng);
        xs[point] = self.range_dist.sample(rng);
    }
}

#[allow(dead_code)]
pub(crate) enum StopCondition<'a, Q: Float + Default> {
    Iterations(usize),
    Timeout(Duration),
    Quality(Q),
    Or(&'a StopCondition<'a, Q>, &'a StopCondition<'a, Q>),
}

pub(crate) trait Terminate {
    type Q;
    fn terminate(&self, iteration: usize, elapsed: &Duration, quality: Self::Q) -> bool;
}

impl<Q> Terminate for StopCondition<'_, Q>
where
    Q: Float + Default,
{
    type Q = Q;

    fn terminate(&self, iteration: usize, elapsed: &Duration, quality: Self::Q) -> bool {
        match self {
            StopCondition::Iterations(max_iters) => iteration >= *max_iters,
            StopCondition::Timeout(limit) => elapsed >= limit,
            StopCondition::Quality(sufficient_quality) => {
                OrderedFloat(quality) <= OrderedFloat(*sufficient_quality)
            }
            StopCondition::Or(a, b) => {
                a.terminate(iteration, elapsed, quality) || b.terminate(iteration, elapsed, quality)
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::alg::stochastic::{Mutate, PointMutation, StopCondition, Terminate, VecMutation};
    use rand::distributions::uniform::Uniform;
    use rand::distributions::Bernoulli;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::hash_map::RandomState;
    use std::collections::HashSet;
    use std::iter::FromIterator;
    use std::time::Duration;

    const INDIVIDUAL: [i32; 3] = [1, 2, 3];
    const NUM_TRIALS: usize = 100;

    #[test]
    fn point_mutation() {
        let n = INDIVIDUAL.len();
        let range: HashSet<&i32, RandomState> = HashSet::from_iter(INDIVIDUAL.iter());
        let mut rng = StdRng::from_entropy();

        let range_dist = Uniform::new_inclusive(0, 3);
        let pm = PointMutation::new(range_dist, n);

        for _ in 0..NUM_TRIALS {
            let ys = pm.copy_mutate(&INDIVIDUAL, &mut rng);
            let num_matches = ys.into_iter().fold(0, |c, y| c + range.contains(&y) as i32) as usize;
            assert!(num_matches >= n - 1 && num_matches <= n);
        }
    }

    #[test]
    fn vec_mutation() {
        let n = INDIVIDUAL.len();
        let mut rng = StdRng::from_entropy();

        let tweak_dist = Bernoulli::from_ratio(1, n as u32).unwrap();
        let range_dist = Uniform::new_inclusive(0, 3);
        let vm = VecMutation::new(tweak_dist, range_dist);

        for _ in 0..NUM_TRIALS {
            let ys = vm.copy_mutate(&INDIVIDUAL, &mut rng);
            assert_eq!(ys.len(), n);
            for y in ys.into_iter() {
                assert!(0 <= y && y <= 3);
            }
        }
    }

    #[test]
    fn termination() {
        let iter_cond = StopCondition::Iterations(1000);
        let time_cond = StopCondition::Timeout(Duration::from_secs(60));
        let quality_cond = StopCondition::Quality(10.);

        let sub_cond = StopCondition::Or(&iter_cond, &time_cond);
        let cond = StopCondition::Or(&quality_cond, &sub_cond);

        assert!(!cond.terminate(0, &Duration::from_secs(10), 11.));
        assert!(cond.terminate(1000, &Duration::from_secs(10), 11.));
        assert!(cond.terminate(0, &Duration::from_secs(60), 11.));
        assert!(cond.terminate(0, &Duration::from_secs(10), 9.));
    }
}
