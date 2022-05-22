use std::cmp::max;

use num_traits::Float;

use crate::Time;

// TODO: Eq is only used in tests, relax T bound by implementing Eq only in tests
#[derive(Debug, PartialEq, Eq)]
pub struct Schedule<T> {
    /// `s[j][k]` is the starting time of the `k`-th part of task `j`
    pub s: Vec<Vec<T>>,
    /// `z[j][k]` is the resource ID on which the `k`-th part of task `j` executes
    pub z: Vec<Vec<usize>>,
    /// c is the makespan `C_max`
    pub c: T,
}

// TODO: try to generalize to `<T: Time + Into<U>, U: Time + Float> -> Option<Schedule<U>>`
//  - potentially define `trait Frac: Time + Float` => `U: Frac`
/// McNaughton's optimal algorithm for `P|pmtn|C_max`
///
/// Runs in `O(n)` worst-case time where `n` is the number of tasks to schedule.
pub fn mcnaughton<T: Time + Float>(p: &[T], r: usize) -> Option<Schedule<T>> {
    if r == 0 {
        return None;
    }

    let c = pmtn_makespan(p.iter().copied(), r)?;

    let n = p.len();

    let mut s = vec![vec![T::zero(); 2]; n];
    let mut z = vec![vec![0; 2]; n];
    let mut p = p.to_vec();
    let mut t = T::zero();
    let mut i = 0;
    let mut j = 0;

    // assign all tasks j to resources i in order, splitting them when makespan c is exceeded
    while j < n {
        if t + p[j] <= c {
            // processing j fits below optimal C_max => use 1st part of its schedule and move on
            s[j][0] = t;
            z[j][0] = i;
            t = t + p[j];
            j += 1;
        } else {
            // use 2nd part of the schedule, update remaining time & move to next resource
            s[j][1] = t;
            z[j][1] = i;
            p[j] = p[j] - (c - t);
            t = T::zero();
            i += 1;
        }
    }

    Some(Schedule { s, z, c })
}

/// Computes [mcnaughton]'s makespan value.
pub(crate) fn pmtn_makespan<T, I>(p: I, r: usize) -> Option<T>
where
    T: Float + Ord,
    I: IntoIterator<Item = T>,
{
    // early return for an empty iter
    let mut p = p.into_iter().peekable();
    p.peek()?;

    // compute max(p) and sum(p) in one go
    let (p_max, p_sum) = p.fold((T::zero(), T::zero()), |(p_max, p_sum), p| {
        (max(p_max, p), p_sum + p)
    });

    // Optimal makespan is computed from two parts that can be interpreted as follows
    //  - `p_max` represents the sequential nature of each task (parts can be assigned to different
    //    resources but cannot run simultaneously)
    //  - `p_sum / r` represents a situation when all resources work without idle waiting
    Some(max(p_max, p_sum / T::from(r)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use itertools::Itertools;
    use ordered_float::OrderedFloat;
    use rstest::*;

    #[rstest]
    #[case(&[], 2)]
    #[case(&[OrderedFloat(4.2)], 0)]
    fn mcnaughton_infeasible(#[case] p: &[OrderedFloat<f64>], #[case] r: usize) {
        assert_eq!(mcnaughton(p, r), None);
    }

    #[rstest]
    #[case::idle(&[10., 8., 4., 14., 1.], 3, 14.)]
    #[case::no_idle(&[2., 3., 2., 3., 2.], 3, 4.)]
    #[case::fractional(&[2., 2., 2., 2., 1.], 4, 2.25)]
    fn mcnaughton_feasible(#[case] p: &[f32], #[case] r: usize, #[case] expected: f32) {
        let p = p.iter().copied().map(OrderedFloat).collect_vec();

        let Schedule {
            s: _,
            z: _,
            c: actual,
        } = mcnaughton(&p, r).expect("feasible solution");

        // TODO: each taks can be processed by at most one resource at a time
        // TODO: each resource can process at most one task at a time
        // TODO: sum of parts of any task is it's processing time

        assert_eq!(OrderedFloat(expected), actual);
    }
}
