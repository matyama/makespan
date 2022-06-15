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
//!
//! ### `P|prec|C_max`
//! TODO
use std::cmp::Reverse;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::{
    cmp::{max, min},
    collections::VecDeque,
};

use crate::mp_pmtn::pmtn_makespan;
use crate::{Prec, Time};

use daggy::{NodeIndex, Walker};
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

/// Search for approx. solution of `P||C_max` using LPT (Longest Processing Time First) algorithm
///
/// ## Input
///  - the processing time `p[j]` of task `j`
///  - the number of parallel identical resources `m`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the resource `z[j]` (in the range `0..m`) that will run task `j`
///  - the optimal makespan `c`
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
///
/// ## Description
/// Asymptotic runtime is `O(n*log(n))` time where `n` is the number of non-preemptive tasks.
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/km` where
///  - `m` is no. resources (`m << n`)
///  - `k` is no. tasks assigned to the resource which finishes last
pub fn lpt<T: Time>(p: &[T], m: usize) -> Option<Schedule<T>> {
    if p.is_empty() || m == 0 {
        return None;
    }

    let n = p.len();

    let mut s = vec![T::zero(); n];
    let mut z = vec![0usize; n];
    let mut c = vec![T::zero(); m];

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

/// Search for optimal solution of `P||C_max` using Branch&Bound
///
/// ## Input
///  - the processing time `p[j]` of task `j`
///  - the number of parallel identical resources `r`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the resource `z[j]` (in the range `0..m`) that will run task `j`
///  - the optimal makespan `c`
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
///
/// ## Branch and Bound implementation
///
/// Finding an optimal schedule for `P || C_max` is proven to be **NP-hard**.
///
/// ### Upper Bound (UB) pruning
/// Initial feasible solution is found by an [`lpt`](crate::mp::lpt) call as set as an UB and then
/// each time a complete solution is found, the UB is updated (if better schedule has been fuound).
///
/// Any schedule whose makespan exceeeds current UB (`C_max >= UB`) during the B&B search is pruned.
///
/// ### Lower Bound (LB) pruning
/// Partial solutions are also lower bounded using a modification of the McNaughton's algorithm for
/// a relaxed problem `P|pmtn|C_max` (i.e. assuming task preeption).
///
/// First, lets denote `C_min(m)` to be the minimum resource completion time assuming a partial
/// solution on `m` resources. Further, let `V` denote the set tasks that remain to be scheduled and
/// `V'` to additionally to `V` also include those already scheduled tasks that continue to run
/// after `C_min(m)` (i.e. start before but finish after). Finally, let `C_pmtn(V')` denote the
/// optimal makespan of an insntace of `P|pmtn|C_max` on tasks in `V'` and the processing time of
/// tasks in `V'\V` to be restricted to the portion that exceeds `C_min(m)`.
///
/// In summary:
///  - `C_min(m) = min_i C_i` where `C_i` is the current maxumum completion time of tasks scheduled
///    to resource `i`
///  - `V'` is a set of tasks that either remain to be scheduled or have not finished until (and
///    including) `C_min(m)`
///  - `C_pmtn(V')` corresponds to the optimal makespan of `P|pmtn|C_max` (found by McNaughton)
///    using tasks `V'` but with processing times of already running times (`V'\V`) reduced to the
///    portion after `C_min(m)`
///
/// Then the LB on a (partial) solution can be computed as `LB = C_min(m) + C_pmtn(V')`. The
/// intuition is that
///  - `C_min(m)` represents the *sequential* characted of already scheduled tasks that is similar
///    to one part of McNaughton's computation of the optimum. Notice that for a complete schedule
///    (no remainig tasks) `C_min(m) <= C_max` of the original problem.
///  - `C_pmtn(V')` is a solution of a relaxed sub-problem in which we allow task preemption - as
///    if we tried to solve `P|pmtn|C_max` with an empty schedule
///  - Tasks that start before but finish after `C_min(m)` are considered sequential in the first
///    part and potentially preemptive in the other part. Hence only the latter is inluded in the
///    relaxation
///  - Because `C_min(m) <= C_max` and `C_pmtn(V')` is a solution to a problem relaxation, it
///    follows that `LB <= C_max`
///
/// Finally, the B&B search prunes any generated node whose `LB >= UB` as further progress cannot
/// lead to an optimum.
///
/// ### Pruning symmetries
/// Since `P||C_max` assumes **identical** resources, their permutation has no effect on the
/// optimal schedule. Once the search reaches certian resource completion times (`C_i`
/// for each resource `i`), the same completion times (and thus makespan `C_max = max_i C_i`) will
/// be yielded in a different portion of the search tree modulo a permutation of the resources
/// (indices `i`).
///
/// The B&B search records hashes of visited states (hashes of sorted completion times `C_i`) and
/// prunes new nodes with states that have already been visited.
///
/// ### Search & heuristics
///  - The search order currently progresses in a depth-first manner and is not directed to
///    specific states
///  - When expanding a node, the search currently always selects the first unscheduled task
///
pub fn bnb<T: Time + Hash + Into<f64>>(p: &[T], m: usize) -> Option<Schedule<T>> {
    if p.is_empty() || m == 0 {
        return None;
    }

    let n = p.len();

    // upper bound solution
    let mut best = lpt(p, m)?;

    // TODO: Best-first BnB
    //  - use a priority queue order by heuristic value (lower bound)
    let mut queue = VecDeque::new();
    queue.push_back(Node::init(m, n));

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

        (0..m)
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
    fn init(m: usize, n: usize) -> Self {
        Self {
            n_tasks: 0,
            s: vec![T::inf(); n],
            z: vec![0; n],
            c: vec![T::zero(); m],
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

/// Search for approx. solution of `P|prec|C_max` using **List Scheduling (LS)**
///
/// ## Input
///  - the processing time `p[j]` of task `j`
///  - the number of parallel identical resources `m`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the resource `z[j]` (in the range `0..m`) that will run task `j`
///  - the optimal makespan `c`
///
/// ## Example
/// TODO
/// ```
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
///
/// ## Description
/// [LS](https://en.wikipedia.org/wiki/List_scheduling) is a greedy algorithm that repeatedly
/// applies following steps:
///  1. TODO
///
/// TODO
/// Asymptotic runtime is `O(n*log(n))` time where `n` is the number of non-preemptive tasks.
///
/// TODO: can be decreased with LPT
/// Approximation factor is `r(LS) = 2 − 1/m` where `m` is no. resources (`m << n`).
///
/// ## Assumptions
///  - [`NodeIndex::index`](NodeIndex::index) corresponds to the indexing of processing times, i.e.
///  node with index `j` in the DAG has processing time `p[j]`
// TODO: remove fmt
pub fn ls<T: Time + std::fmt::Debug>(prec: Prec, p: &[T], m: usize) -> Option<Schedule<T>> {
    if prec.node_count() == 0 || p.is_empty() || prec.node_count() != p.len() || m == 0 {
        return None;
    }

    let n = p.len();

    // t[i] represents the free time resource i
    let mut t = vec![T::zero(); m];

    let mut s = vec![T::zero(); n];
    let mut z = vec![0; n];
    let mut c = T::zero();

    // init the list using LPT
    let list = p
        .iter()
        .enumerate()
        .sorted_by_key(|(_, &p)| Reverse(p))
        .map(|(j, _)| j)
        .collect_vec();

    // init the list in arbitrary order
    // let list = (0..n).collect_vec();

    // availability a[j] = {
    //     t[k]                      if j has no preds
    //     inf                       if at least one j's pred is in the list
    //     max_j' { s[j'] + p[j'] }  if all j's predecessors j' have been removed
    // }
    let mut av = (0..n)
        .map(|j| Availability::<T>::init(j, &prec))
        .collect_vec();

    let mut removed = vec![false; n];

    // TODO: this is O(n^2) => can we do better?
    //  - to achieve O(n*log(n)) we need O(log(n)) remove and find min available task
    //  - worst-case |children| is in O(n) => still worst-case O(n^2)
    //  - but for sufficiently sparse DAGs it could be better
    for _ in 0..n {
        // choose the resource with the the lowest free time
        let k = t.iter().position_min()?;

        // TODO: the problem with this is that we always walk (and filter) the whole list even
        //       through it should shrink by one each iteration
        //
        // find task in the list with the smallest availability time (breaking ties by the list ord)
        let (j, a) = list
            .iter()
            // TODO: or physically remove from the list?
            //  - not quite good with `Vec` => maybe tree set (?) => O(n*log(n))
            .copied()
            .filter(|&j| !removed[j])
            .map(|j| (j, av[j].value(t[k])))
            .min_by_key(|(_, a)| *a)
            .expect("at least one remainig available task");

        // remove j from the list
        removed[j] = true;

        // assign task j to resource k
        s[j] = max(t[k], a);
        z[j] = k;

        t[k] = s[j] + p[j];
        c = max(c, t[k]);

        prec.children(NodeIndex::new(j))
            .iter(&prec)
            .map(|(_, n)| n.index())
            .for_each(|child| {
                if let Availability::PredAcc { remainig, c_max } = av[child] {
                    av[child] = Availability::PredAcc {
                        remainig: remainig - 1,
                        c_max: max(c_max, t[k]), // NOTE: updated `t[k] = s[j] + p[j]`
                    }
                }
            });
    }

    Some(Schedule { s, z, c })
}

enum Availability<T> {
    // TODO: isn't NoPred the same as `remainig == 0` => yes, except `c_max = t[k]` instead of zero
    NoPred,
    PredAcc { remainig: usize, c_max: T },
}

impl<T: Time> Availability<T> {
    fn init(j: usize, prec: &Prec) -> Self {
        let remainig = prec.parents(NodeIndex::new(j)).iter(prec).count();
        if remainig == 0 {
            Self::NoPred
        } else {
            Self::PredAcc {
                remainig,
                c_max: T::zero(),
            }
        }
    }

    #[inline]
    fn value(&self, c_min: T) -> T {
        match self {
            Self::NoPred => c_min,
            &Self::PredAcc { remainig, c_max } if remainig == 0 => c_max,
            _ => T::inf(),
        }
    }
}

// TODO: use or remove
pub trait TaskList {
    fn list<T: Time>(self, p: &[T]) -> Vec<usize>;
}

#[allow(dead_code, clippy::upper_case_acronyms)]
enum LS {
    Unordered,
    LPT,
}

impl TaskList for LS {
    fn list<T: Time>(self, p: &[T]) -> Vec<usize> {
        match self {
            Self::Unordered => (0..p.len()).collect_vec(),
            Self::LPT => p
                .iter()
                .enumerate()
                .sorted_by_key(|(_, &p)| Reverse(p))
                .map(|(j, _)| j)
                .collect_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daggy::Dag;
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

    #[test]
    fn ls_feasible() {
        // TODO
        let _expected: Schedule<u8> = Schedule {
            s: vec![0, 0, 4, 0, 8, 5, 5],
            z: vec![1, 0, 2, 2, 0, 1, 0],
            c: 11,
        };

        let edges = [(0, 6), (0, 7), (2, 3), (2, 4), (2, 5)];
        let prec = Dag::<(), ()>::from_edges(edges).expect("DAG");

        let p: &[u8] = &[3, 4, 2, 4, 4, 2, 13, 2];

        let actual = ls(prec, p, 2).expect("feasible solution");
        //assert_eq!(expected, actual);

        //assert_eq!(actual.c, 17); // L unsorted
        assert_eq!(actual.c, 18); // L with LPT
    }

    #[rstest]
    #[case(&[], &[4, 2])]
    #[case(&[(0, 1)], &[])]
    fn ls_infeasible(#[case] edges: &[(u32, u32)], #[case] p: &[u8]) {
        let prec = Dag::<(), ()>::from_edges(edges).expect("DAG");
        assert_eq!(ls(prec, p, 2), None);
    }
}
