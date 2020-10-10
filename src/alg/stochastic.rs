use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Bernoulli, Uniform};
use rand::prelude::*;
use std::marker::PhantomData;

pub trait Mutate<T, D>
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
pub struct VecMutation<T, D>
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
        for i in 0..xs.len() {
            if self.tweak_dist.sample(rng) {
                xs[i] = self.range_dist.sample(rng);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct PointMutation<T, D>
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

#[cfg(test)]
mod test {
    use crate::alg::stochastic::{Mutate, PointMutation, VecMutation};
    use rand::distributions::uniform::Uniform;
    use rand::distributions::Bernoulli;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::collections::hash_map::RandomState;
    use std::collections::HashSet;
    use std::iter::FromIterator;

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
}
