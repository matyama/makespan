//! # Non-preemptive single-processor scheduling
//! This module is dedicated to **single-processor** schduling problems **without task preeption**.
//!
//! ## Additional assumptions
//!  - There is only one processor unit(resource)
//!  - Once tasks are started, they cannot be interrupted (non-preemptive)
//!
//! ## Scheduling problems
//!
//! ### `1||C_max`
//! The `1||C_max` problem is described for *single resource* with *no task constraints* and can be
//! solved in linear time by [`sp::unconstrained`](crate::sp::unconstrained).
//!
//! ### `1|prec|C_max`
//! The `1|prec|C_max` problem is described for *single resource* with *task precedence constraints*
//! and can be solved in linear time by [`sp::schedule_prec`](crate::sp::schedule_prec).
//!
//! The task precedence relation is given as a DAG [`Prec`](Prec).
//!
//! ### `1|r_j|C_max`
//! The `1|r_j|C_max` problem is described for *single resource* with *task realease time
//! constraints* and can be solved in log-linear time by
//! [`sp::schedule_release`](crate::sp::schedule_release).
//!
//! The task release times are given as a slice of the same length as the processing times, i.e.
//! `r[j]`/`p[j]` is the release/processing time of task `j`.
//!
//! ### `1|d'_j|C_max`
//! The `1|d'_j|C_max` problem is described for *single resource* with *task deadline constraints*
//! and can be solved in log-linear time by [`sp::edf`](crate::sp::edf).
//!
//! The task (hard) deadlines are given as a slice of the same length as the processing times, i.e.
//! `d[j]` is the deadline of task `j` with corresponding processing time `p[j]`.
//!
//! ### `1|r_j,d'_j|C_max`
//! The `1|r_j,d'_j|C_max` problem is described for *single resource* with both *task deadline* and
//! *task relase time* constraints and is known to be **NP-hard**.
//!
//! There are two algorithms solving this problem:
//!  1. A heuristic algorithm called **Earliest Deadline First (EDF)** can be found in
//!     [`sp::release_deadline`](crate::sp::release_deadline)
//!  1. TODO: Bratley's optimal *Branch&Bound (BnB)* algorithm `sp::bratley`
use std::cmp::max;

use crate::{Prec, Time};

use daggy::petgraph::algo::toposort;
use daggy::NodeIndex;
use itertools::Itertools;

/// Single resource schedule and corresponding makespan value
#[derive(Debug, PartialEq, Eq)]
pub struct Schedule<T> {
    /// the starting time `s[j]` for each task `j`
    pub s: Vec<T>,
    /// maximum completion time (a.k.a the **makespan**)
    pub c: T,
}

/// Linear algorithm for the unconstrained problem `1||C_max`.
///
/// ## Input
/// The processing time `p[j]` of each task `j`.
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the optimal makespan `c`
///
/// ## Description
/// Note that although tasks can be scheduled in an arbitrary order, this implementation actually
/// processes them *in-order*.
///
/// Runs in `O(n)` worst-case time where `n` is the number of tasks to schedule.
///
/// ## Example
/// ```
/// # extern crate makespan;
/// use makespan::sp;
///
/// // Define a vector of processing times for each task
/// let p: Vec<u8> = vec![1, 2];
///
/// // Find an optimal schedule
/// let schedule = sp::unconstrained(&p)
///     .expect("feasible solution");
///
/// assert_eq!(schedule, sp::Schedule { s: vec![0, 1], c: 3 });
/// ```
pub fn unconstrained<T: Time>(p: &[T]) -> Option<Schedule<T>> {
    if p.is_empty() {
        return None;
    }

    let n = p.len();

    let mut s = vec![T::zero(); n];
    for i in 0..(n - 1) {
        s[i + 1] = s[i] + p[i];
    }

    let c = s[n - 1] + p[n - 1];

    Some(Schedule { s, c })
}

// TODO: Is it ok to relay on the fact that `NodeIndex::index` aligns with `p` indexing?
//  - then it would be safer to have `p` as node weights
/// Linear algorithm for `1|prec|C_max`.
///
/// ## Input
///  - [`daggy::Dag<(), ()>`](daggy::Dag) representing task precedence relation
///  - the processing time `p[j]` of task `j`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the optimal makespan `c`
///
/// ## Description
/// The algorithm schedules tasks in the *topological order* of their precedence relation DAG.
///
/// Runs in `O(n)` worst-case time where `n` is the number of tasks to schedule.
///
/// ## Assumptions
///  - [`NodeIndex::index`](NodeIndex::index) corresponds to the indexing of processing times, i.e.
///  node with index `j` in the DAG has processing time `p[j]`
///
/// ## Example
/// ```
/// # extern crate makespan;
/// use makespan::sp;
/// use daggy::Dag;
///
/// // Define a task precedence DAG
/// let edges: &[(u32, u32)] = &[(0, 1), (0, 2), (1, 3), (2, 3)];
/// let prec = Dag::<(), ()>::from_edges(edges)
///     .expect("DAG");
///
/// // Define processing time for each task
/// let p: &[u8] = &[2, 1, 3, 1];
///
/// // Find an optimal schedule given the precedence relation
/// let schedule = sp::schedule_prec(&prec, p)
///     .expect("feasible solution");
///
/// assert_eq!(schedule, sp::Schedule { s: vec![0, 5, 2, 6], c: 7 });
/// ```
pub fn schedule_prec<T: Time>(prec: &Prec, p: &[T]) -> Option<Schedule<T>> {
    if prec.node_count() == 0 || p.is_empty() || prec.node_count() != p.len() {
        return None;
    }

    let order = toposort(prec, None).expect("DAGs do not have cycles");

    let mut c = T::zero();
    let mut s = vec![T::zero(); p.len()];

    for i in order.into_iter().map(NodeIndex::index) {
        s[i] = c;
        c = c + p[i];
    }

    Some(Schedule { s, c })
}

/// Algorithm for `1|r_j|C_max`.
///
/// ## Input
///  - the release time `r[j]` of task `j`
///  - the processing time `p[j]` of task `j`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the optimal makespan `c`
///
/// ## Description
/// The algorithm schedules tasks in a non-decreasing order of their processing times.
///
/// Runs in `O(n*log(n))` worst-case time where `n` is the number of tasks to schedule.
///
/// ## Example
/// ```
/// # extern crate makespan;
/// use makespan::sp;
///
/// // Define release times for each task
/// let r: &[u8] = &[1, 3];
///
/// // Define processing times for each task
/// let p: &[u8] = &[2, 1];
///
/// // Find an optimal schedule
/// let schedule = sp::schedule_release(&r, &p)
///     .expect("feasible solution");
///
/// assert_eq!(schedule, sp::Schedule { s: vec![1, 3], c: 4 });
/// ```
pub fn schedule_release<T: Time>(r: &[T], p: &[T]) -> Option<Schedule<T>> {
    if r.is_empty() || p.is_empty() || r.len() != p.len() {
        return None;
    }

    let n = r.len();

    // sort tasks in non-decreasing order of their relaese time
    let tasks = r.iter().enumerate().sorted_by_key(|(_, &r)| r);

    let mut c = T::zero();
    let mut s = vec![T::zero(); n];

    for (i, &r) in tasks {
        // last completion time is the next task's start time unless we need to postpone till
        // task's release time
        s[i] = max(c, r);
        c = s[i] + p[i];
    }

    Some(Schedule { s, c })
}

/// Earliest Deadline First (EDF) algorithm for `1|d'_j|C_max`.
///
/// ## Input
///  - the (hard) deadline `d[j]` of task `j`
///  - the processing time `p[j]` of task `j`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the optimal makespan `c`
///
/// ## Description
/// The algorithm attempts to schedule times in non-decreasing order of their deadlines `d[j]`.
///
/// Note that not all instances (i.e. combinations of inputs `d` and `p`) can yield feasible
/// solution.
///
/// Runs in `O(n*log(n))` where `n` is the number of tasks to schedule.
///
/// ## Example: feasible solution
/// ```
/// # extern crate makespan;
/// use makespan::sp;
///
/// // Define deadlines for each task
/// let d: &[u8] = &[4, 6, 3];
///
/// // Define processing times for each task
/// let p: &[u8] = &[1, 2, 3];
///
/// // Find an optimal schedule
/// let schedule = sp::edf(&d, &p)
///     .expect("feasible solution");
///
/// assert_eq!(schedule, sp::Schedule { s: vec![3, 4, 0], c: 6 });
/// ```
///
/// ## Example: infeasible instance
/// ```should_panic
/// # extern crate makespan;
/// use makespan::sp;
///
/// // Define deadlines for each task
/// let d: &[u8] = &[3, 6, 3];
///
/// // Define processing times for each task
/// let p: &[u8] = &[1, 1, 3];
///
/// // Find this instance infeasible
/// let _ = sp::edf(&d, &p)
///     .expect("feasible solution");
/// ```
pub fn edf<T: Time>(d: &[T], p: &[T]) -> Option<Schedule<T>> {
    if d.is_empty() || p.is_empty() || d.len() != p.len() {
        return None;
    }

    let n = d.len();

    // TODO: use O(n) sort => can't use itertools
    // sort tasks in non-decreasing order of their deadline
    let tasks = d.iter().enumerate().sorted_by_key(|(_, &d)| d);

    let mut c = T::zero();
    let mut s = vec![T::zero(); n];

    for (i, &d) in tasks {
        // unfeasible solution: passed task's deadline
        if c + p[i] > d {
            return None;
        }
        // last completion time is the next task's start time
        s[i] = c;
        c = c + p[i];
    }

    Some(Schedule { s, c })
}

/// Heuristic solution (sub-optimal) to `1|r_j,d'_j|C_max`.
///
/// ## Input
///  - the release time `r[j]` of task `j`
///  - the (hard) deadline `d[j]` of task `j`
///  - the processing time `p[j]` of task `j`
///
/// ## Output
/// The resulting [Schedule] consists of the following parts:
///  - the starting time `s[j]` for each task `j`
///  - the optimal makespan `c`
///
/// ## Description
/// Adapts EDF to which it adds relaease time constrants and runs in `O(n*log(n))` worst-case
/// time where `n` is the number of tasks to schedule.
///
/// Note that similarly to EDF, not all instances are feasible due to deadlines.
pub fn release_deadline<T: Time>(r: &[T], d: &[T], p: &[T]) -> Option<Schedule<T>> {
    schedule_release_deadline(r, d, p).map(Schedule::from)
}

fn schedule_release_deadline<T: Time>(r: &[T], d: &[T], p: &[T]) -> Option<TaskSchedule<T>> {
    if r.is_empty() || d.is_empty() || p.is_empty() {
        return None;
    }

    let n = p.len();

    if r.len() != n || d.len() != n {
        return None;
    }

    // first take tasks in non-decreasing order of deadlines and those that are equal are processed
    // in non-decreasing order of their release times
    let tasks = d
        .iter()
        .zip(r)
        .enumerate()
        .sorted_by_key(|(_, (&d, &r))| (d, r));

    let mut c = T::zero();
    let mut s = vec![T::zero(); n];
    let mut t = vec![0; n];

    for (k, (i, (&d, &r))) in tasks.enumerate() {
        t[k] = i;
        s[i] = max(c, r);
        c = s[i] + p[i];
        if c > d {
            return None;
        }
    }

    Some(TaskSchedule { t, s, c })
}

struct TaskSchedule<T> {
    #[allow(dead_code)]
    t: Vec<usize>,
    s: Vec<T>,
    c: T,
}

impl<T> From<TaskSchedule<T>> for Schedule<T> {
    fn from(x: TaskSchedule<T>) -> Self {
        Self { s: x.s, c: x.c }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use daggy::Dag;
    use rstest::*;

    #[rstest]
    #[case(&[1], Schedule { s: vec![0], c: 1 })]
    #[case(&[1, 2], Schedule { s: vec![0, 1], c: 3 })]
    fn unconstrained_feasible(#[case] p: &[u8], #[case] expected: Schedule<u8>) {
        let actual = unconstrained(p).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[test]
    fn unconstrained_infeasible() {
        let p: &[u8] = &[];
        assert_eq!(unconstrained(p), None);
    }

    #[rstest]
    #[case(&[(0, 1)], &[1, 2], Schedule { s: vec![0, 1], c: 3 })]
    #[case(&[(0, 1), (0, 2), (1, 2)], &[1, 2, 1], Schedule { s: vec![0, 1, 3], c: 4 })]
    #[case(
        &[(0, 1), (0, 2), (1, 3), (2, 3)],
        &[2, 1, 3, 1],
        Schedule { s: vec![0, 5, 2, 6], c: 7 }
    )]
    fn prec_feasible(
        #[case] edges: &[(u32, u32)],
        #[case] p: &[u8],
        #[case] expected: Schedule<u8>,
    ) {
        let prec = Dag::<(), ()>::from_edges(edges).expect("DAG");
        let actual = schedule_prec(&prec, p).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[test]
    fn prec_isolated() {
        let mut prec = Dag::new();
        prec.add_node(());
        prec.add_node(());
        prec.add_node(());

        let p: &[u8] = &[1, 1, 1];

        let Schedule { mut s, c } = schedule_prec(&prec, p).expect("feasible solution");

        // orderdering is in this case irrelevant since there are no precedences and processing
        // times are uniform
        s.sort();
        assert_eq!(s, vec![0, 1, 2]);
        assert_eq!(c, 3);
    }

    #[rstest]
    #[case(&[], &[4, 2])]
    #[case(&[(0, 1)], &[])]
    fn prec_infeasible(#[case] edges: &[(u32, u32)], #[case] p: &[u8]) {
        let prec = Dag::<(), ()>::from_edges(edges).expect("DAG");
        assert_eq!(schedule_prec(&prec, p), None);
    }

    #[rstest]
    #[case(&[0], &[2], Schedule { s: vec![0], c: 2 })]
    #[case(&[1], &[2], Schedule { s: vec![1], c: 3 })]
    #[case(&[1, 3], &[2, 1], Schedule { s: vec![1, 3], c: 4 })]
    #[case(&[4, 1], &[1, 2], Schedule { s: vec![4, 1], c: 5 })]
    #[case(&[2, 1], &[1, 2], Schedule { s: vec![3, 1], c: 4 })]
    fn release_feasible(#[case] r: &[u8], #[case] p: &[u8], #[case] expected: Schedule<u8>) {
        let actual = schedule_release(r, p).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(&[], &[])]
    #[case(&[1], &[])]
    #[case(&[], &[1])]
    #[case(&[1], &[1, 2])]
    #[case(&[1, 2], &[1])]
    fn release_infeasible(#[case] r: &[u8], #[case] p: &[u8]) {
        assert_eq!(schedule_release(r, p), None);
    }

    #[rstest]
    #[case(&[4], &[2], Schedule { s: vec![0], c: 2 })]
    #[case(&[3, 3], &[1, 1], Schedule { s: vec![0, 1], c: 2 })]
    #[case(&[3, 5], &[1, 4], Schedule { s: vec![0, 1], c: 5 })]
    #[case(&[4, 6, 3], &[1, 2, 3], Schedule { s: vec![3, 4, 0], c: 6 })]
    fn edf_feasible(#[case] d: &[u8], #[case] p: &[u8], #[case] expected: Schedule<u8>) {
        let actual = edf(d, p).expect("feasible solution");
        assert_eq!(expected, actual);
    }

    #[rstest]
    #[case(&[], &[])]
    #[case(&[1], &[])]
    #[case(&[], &[1])]
    #[case(&[1], &[2])]
    #[case(&[3, 6, 3], &[1, 1, 3])]
    fn edf_infeasible(#[case] d: &[u8], #[case] p: &[u8]) {
        assert_eq!(edf(d, p), None);
    }
}
