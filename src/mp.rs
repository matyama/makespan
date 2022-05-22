//! # Non-preemptive multi-processor scheduling
//! This module is dedicated to **multi-processor** schduling problems **without task preeption**.
//!
//! ## Additional assumptions
//!  - There are multiple processing units (resources)
//!  - Once tasks are started, they cannot be interrupted (non-preemptive)
//!
//! ## Scheduling problems
//!
//! ### `P||C_max`
//! The `P||C_max` problem is described for *parallel identical resources* with *no task
//! constraints* and is proven to be *NP-hard*.
//!
//! There are two implementations solving this problem:
//!  1. An approximate algorithm called **Longest Processing Time First (LPT)** can be found in
//!     [`mp::lpt`](src/mp.rs)
//!  1. An optimal *Branch&Bound (BnB)* algorithm [`mp::bnb`](src/mp.rs)
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::{
    cmp::{max, min},
    collections::VecDeque,
};

use crate::mp_pmtn::pmtn_makespan;
use crate::Time;

use itertools::{Itertools, MinMaxResult};
use ordered_float::OrderedFloat;

// TODO: Eq is only relevant in tests, relax T bound and implement it just there
/// Parallel resouce allocation and schedule with corresponding makespan value
#[derive(Debug, PartialEq, Eq)]
pub struct Schedule<T: Eq> {
    /// the starting time `s[j]` for each task `j`
    pub s: Vec<T>,
    /// the resource `z[j]` (in the range `0..r`) that will run task `j`
    pub z: Vec<usize>,
    /// maximum completion time across all resources (a.k.a the **makespan**)
    pub c: T,
}

/// Search for approx. solution of `P || C_max` using LPT (Longest Processing Time First) algorithm
/// given
///  - the processing time `p[j]` of task `j`
///  - the number of parallel identical resources `r`
///
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the resource `z[j]` (in the range `0..r`) that will run task `j`
///  - the optimal makespan `c`
///
/// Asymptotic runtime is `O(n*log(n))` time where `n` is the number of non-preemptive tasks.
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/kR` where
///  - `R` is no. resources (`R << n`)
///  - `k` is no. tasks assigned to the resource which finishes last
///
/// ## Example
/// ```rust
/// # extern crate makespan;
/// use makespan::mp;
///
/// // Define a vector of processing times for each task
/// let p: Vec<u8> = vec![5, 5, 4, 4, 3, 3, 3];
///
/// // Find approximate schedule for 3 resources
/// let mp::Schedule { s: _, z: _, c } = mp::lpt(&p, 3)
///     .expect("feasible solution");
///
/// assert_eq!(c, 11);
/// ```
pub fn lpt<T: Time>(p: &[T], r: usize) -> Option<Schedule<T>> {
    if p.is_empty() || r == 0 {
        return None;
    }

    let n = p.len();

    let mut s = vec![T::zero(); n];
    let mut z = vec![0usize; n];
    let mut c = vec![T::zero(); r];

    // TODO: consider linear sort for larger inputs as this is the main complexity component
    let tasks = p.iter().enumerate().sorted_by_key(|(_, p)| *p).rev();

    for (i, &p) in tasks {
        let j_min = c.iter().map(|&c| c + p).position_min()?;
        s[i] = c[j_min];
        z[i] = j_min;
        c[j_min] = c[j_min] + p;
    }

    // TODO: max should always exist for n > 0
    let c = c.into_iter().max()?;

    Some(Schedule { s, z, c })
}

// TODO: document `LB = C_min(R) + C_pmtn(V')` where
//  - `C_min(R) = min_i c_i` is the min. completion time across currently scheduled resources
//  - `C_pmtn(V')` is the opt (McNaughton) makespan of `P|pmtn|C_max` with tasks from a set V'
//  - `V = {unscheduled tasks}`
//  - `V' = V + {portion of scheduled tasks which runs after C_min(R)}`
// I.e. LB is "min resouce completion time" + "McNaughton with all processing time after C_min(R)".
/// Search for optimal solution of `P || C_max` using Branch&Bound given
///  - the processing time `p[j]` of task `j`
///  - the number of parallel identical resources `r`
///
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the resource `z[j]` (in the range `0..r`) that will run task `j`
///  - the optimal makespan `c`
///
/// Note that finding an optimal schedule for `P || C_max` is proven to be *NP-hard*.
///
/// ## Example
/// ```rust
/// # extern crate makespan;
/// use makespan::mp;
///
/// // Define a vector of processing times for each task
/// let p: Vec<u8> = vec![5, 5, 4, 4, 3, 3, 3];
///
/// // Find optimal schedule for 3 resources
/// let mp::Schedule { s: _, z: _, c } = mp::bnb(&p, 3)
///     .expect("feasible solution");
///
/// assert_eq!(c, 9);
/// ```
pub fn bnb<T: Time + Hash + Into<f64>>(p: &[T], r: usize) -> Option<Schedule<T>> {
    if p.is_empty() || r == 0 {
        return None;
    }

    let n = p.len();

    // upper bound solution
    let mut best = lpt(p, r)?;

    // TODO: Best-first BnB
    //  - use a priority queue order by heuristic value (lower bound)
    let mut queue = VecDeque::new();
    queue.push_back(Node::init(n, r));

    // set of visited states to prune symmetries
    let mut visited = HashSet::new();

    while let Some(node) = queue.pop_front() {
        // check if node represents a complete solution
        if node.n_tasks == n {
            // update best solution found so far
            if node.c_max < best.c {
                best = node.into();
            }
            continue;
        }

        // TODO: better heuristic
        // heuristic: select first uncheduled task
        let j = (0..n)
            .find(|&j| node.s[j].is_inf())
            .expect("partial solution should have unscheudled tasks");

        (0..r)
            .filter_map(|i| node.expand(i, j, p, best.c))
            .for_each(|n| {
                let state = n.state();
                if !visited.contains(&state) {
                    visited.insert(state);
                    queue.push_front(n);
                }
            });
    }

    Some(best)
}

struct Node<T> {
    /// current number of scheduled tasks
    n_tasks: usize,
    /// `s[j]` is the starting time of task `j` (`s[j] = inf` iff `j` is not scheduled yet)
    s: Vec<T>,
    /// `z[j] = i` iff task `j` is allocated to resource `i`
    z: Vec<usize>,
    /// `c[i]` is the maximum completion time at resouce `i`
    c: Vec<T>,
    /// current makespan `c_max = max_i c[i]`
    c_max: T,
}

impl<T: Time + Into<f64>> Node<T> {
    fn init(n: usize, r: usize) -> Self {
        Self {
            n_tasks: 0,
            s: vec![T::inf(); n],
            z: vec![0; n],
            c: vec![T::zero(); r],
            c_max: T::zero(),
        }
    }

    /// Expand this [Node] by adding `j -> i` to the schedule where
    ///  - `j` is new (unscheduled) task
    ///  - `i` is the resource that will run `j`
    fn expand(&self, i: usize, j: usize, p: &[T], ub: T) -> Option<Self> {
        let mut s = self.s.to_vec();
        s[j] = self.c[i];

        let mut z = self.z.to_vec();
        z[j] = i;

        // NOTE: can be inferred from s and z => potentially could be omitted from [Node]
        let mut c = self.c.to_vec();
        c[i] = c[i] + p[j];

        let c_max = max(self.c_max, c[i]);

        if c_max >= ub {
            return None;
        }

        let lb = bnb_lb(&s, p, &c);

        if lb >= ub.into() {
            return None;
        }

        Some(Self {
            n_tasks: self.n_tasks + 1,
            s,
            z,
            c,
            c_max,
        })
    }
}

impl<T: Time + Hash> Node<T> {
    /// State is represented as a hash of sorted max. completion times at each resource
    fn state(&self) -> u64 {
        // TODO: use [aHash](https://crates.io/crates/ahash)
        let mut hasher = DefaultHasher::new();
        let mut state = self.c.to_vec();
        state.sort();
        state.hash(&mut hasher);
        hasher.finish()
    }
}

impl<T: Time> From<Node<T>> for Schedule<T> {
    fn from(n: Node<T>) -> Self {
        Self {
            s: n.s,
            z: n.z,
            c: n.c_max,
        }
    }
}

fn bnb_lb<T: Time + Into<f64>>(s: &[T], p: &[T], c: &[T]) -> f64 {
    let (c_min, c_max) = match c.iter().minmax() {
        MinMaxResult::NoElements => (T::zero(), T::zero()),
        MinMaxResult::OneElement(&m) => (m, m),
        MinMaxResult::MinMax(&c_min, &c_max) => (c_min, c_max),
    };

    let p_pmtn = s
        .iter()
        .zip(p)
        .filter_map(|(&s, &p)| {
            // FIXME: inf is not a representative values for int types => e.g. cannot distinguish
            // u8::MAX from p[j] = 255 => cannot tell if j has not yet been scheduled
            if s.is_inf() {
                Some(p.into())
            } else {
                let c = s + p;
                if c > c_min {
                    Some(min(c - c_min, p).into())
                } else {
                    None
                }
            }
        })
        .map(OrderedFloat);

    // compute relaxed scheduling problem `P|pmtn|C_max` on `p_pmtn`
    if let Some(c_pmtn) = pmtn_makespan(p_pmtn, c.len()) {
        c_min.into() + c_pmtn.into_inner()
    } else {
        c_max.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;
    use rstest::*;

    #[test]
    fn lpt_feasible() {
        // note: other equally valued symmetries are also valid solutions
        let expected: Schedule<u8> = Schedule {
            s: vec![0, 0, 4, 0, 8, 5, 5],
            z: vec![1, 0, 2, 2, 0, 1, 0],
            c: 11,
        };

        let p = vec![5, 5, 4, 4, 3, 3, 3];

        let actual = lpt(&p, 3).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(&[], 2)]
    #[case(&[OrderedFloat(4.2)], 0)]
    fn lpt_infeasible(#[case] p: &[OrderedFloat<f64>], #[case] r: usize) {
        assert_eq!(lpt(p, r), None);
    }

    #[test]
    fn bnb_feasible() {
        // note: other equally valued symmetries are also valid solutions
        let expected: Schedule<u8> = Schedule {
            s: vec![0, 0, 5, 5, 0, 3, 6],
            z: vec![0, 1, 0, 1, 2, 2, 2],
            c: 9,
        };

        let p = vec![5, 5, 4, 4, 3, 3, 3];

        let actual = bnb(&p, 3).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[test]
    fn bnb_feasible2() {
        // note: other equally valued symmetries are also valid solutions
        let expected: Schedule<u8> = Schedule {
            s: vec![3, 0, 2, 0, 0],
            z: vec![0, 1, 2, 0, 2],
            c: 5,
        };

        let p = vec![2, 3, 2, 3, 2];

        let actual = bnb(&p, 3).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(&[], 2)]
    #[case(&[OrderedFloat(4.2)], 0)]
    fn bnb_infeasible(#[case] p: &[OrderedFloat<f64>], #[case] r: usize) {
        assert_eq!(bnb(p, r), None);
    }
}
