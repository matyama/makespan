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
//! The `1||C_max` problem is described for *single resource* with *no task
//! constraints* and can be solved in linear time by [`sp::unconstrained`](src/sp.rs).
//!
//! ### `1|prec|C_max`
//!
//! ### `1|r_j|C_max`
//!
//! ### `1|d'_j|C_max`
//!
//! ### `1|r_j,d'_j|C_max`
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

/// Linear algorithm for `1||C_max`.
///
/// Runs in `O(n)` worst-case time where `n` is the number of tasks to schedule.
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
/// Algorithm for `1|prec|C_max`
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
/// Runs in `O(n*log(n))` worst-case time where `n` is the number of tasks to schedule.
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
/// Runs in `O(n*log(n))` where `n` is the number of tasks to schedule.
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
/// Adapts EDF to which it adds relaease time constrants and runs in `O(n*log(n))` worst-case
/// time where `n` is the number of tasks to schedule.
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
        // TODO: is this sound?
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
