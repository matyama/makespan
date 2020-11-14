use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::iter::Sum;
use std::ops::RangeInclusive;
use std::time::Duration;

use num_traits::Float;
use ordered_float::OrderedFloat;
use voracious_radix_sort::Radixable;

/// Data structure representing an input scheduling task
///
/// A task consist of two components:
///  - unique ID
///  - processing time
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Task<T: Float> {
    /// Unique task ID
    pub(crate) id: usize,
    /// Processing time (should be a positive float)
    pub(crate) pt: OrderedFloat<T>,
}

impl<T: Float> Eq for Task<T> {}

impl<T: Float> PartialOrd for Task<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for Task<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.pt.cmp(&other.pt)
    }
}

impl<T: Float> Radixable<f64> for Task<T> {
    type Key = f64;
    #[inline]
    fn key(&self) -> Self::Key {
        // TODO: handle unwrap safely
        // TODO: this is probably quite slow
        self.pt.to_f64().expect("Cannot convert to f64")
    }
}

/// Data structure holding resulting `schedule` and objective `value`.
///
/// The schedule is represented as a vector of resources which is indexed by tasks. Additionally,
/// this structure records the value of the schedule (maximum completion time) and original number
/// of resources.
#[derive(Debug)]
pub struct Solution<T: Float> {
    /// assignment: task j -> resource i (i.e. `schedule[j] = i`)
    pub schedule: Vec<usize>,
    /// total makespan (max completion time over all resources)
    pub value: T,
    /// number of resources
    pub num_resources: usize,
}

impl<T: Float> Solution<T> {
    /// Generate schedule in the form of a Gantt chart, i.e. as mapping: `resource -> [tasks]`.
    ///
    /// # Example
    /// ```rust
    /// # extern crate makespan;
    /// let solution = makespan::Solution {
    ///     schedule: vec![0, 1, 0, 2], value: 2., num_resources: 3,
    /// };
    /// let gantt = solution.gantt_schedule();
    /// assert_eq!(gantt[&0], vec![0, 2]);
    /// assert_eq!(gantt[&1], vec![1]);
    /// assert_eq!(gantt[&2], vec![3]);
    /// ```
    pub fn gantt_schedule(&self) -> HashMap<usize, Vec<usize>> {
        let mut gantt: HashMap<usize, Vec<usize>> = HashMap::with_capacity(self.num_resources);

        for (task, &resource) in self.schedule.iter().enumerate() {
            match gantt.entry(resource) {
                Entry::Occupied(mut e) => e.get_mut().push(task),
                Entry::Vacant(e) => {
                    e.insert(vec![task]);
                }
            }
        }

        gantt
    }

    /// Compute distribution of tasks to resources, i.e. `loads[resource] = #tasks`
    ///
    /// # Example
    /// ```rust
    /// # extern crate makespan;
    /// let solution = makespan::Solution {
    ///     schedule: vec![0, 1, 0, 2], value: 2., num_resources: 3,
    /// };
    /// assert_eq!(solution.task_loads(), vec![2, 1, 1]);
    /// ```
    pub fn task_loads(&self) -> Vec<u32> {
        let mut task_dist = vec![0u32; self.num_resources];
        for &resource in self.schedule.iter() {
            task_dist[resource] += 1;
        }
        task_dist
    }
}

// TODO: maybe use as a return type for the schedule?
//  - however, API might not be that nice because once would have to check the instance even though
//  - just one would be expected
// pub enum Solution {
//     TASKS(Vec<usize>),
//     GANTT(HashMap<usize, Vec<usize>>),
// }

/// Data structure that contains various statistics collected during the scheduling.
#[derive(Debug)]
pub struct Stats<T: Float> {
    /// total makespan (max completion time over all resources)
    pub value: T,
    /// approximation factor
    pub approx_factor: f64,
    /// number of resources
    pub num_resources: usize,
    /// number of tasks
    pub num_tasks: usize,
    /// elapsed time in since scheduling started
    pub elapsed: Duration,
    /// no. expanded nodes
    pub expanded: u64,
    /// no. nodes pruned based on objective value
    pub pruned_value: u64,
    /// no. nodes pruned based on heuristic value
    pub pruned_h_value: u64,
    /// no. nodes pruned because state was re-visited
    pub pruned_closed: u64,
    /// true iff optimal solution has been found within time limit
    pub proved_optimal: bool,
}

impl<T: Float> Stats<T> {
    /// Create new stats for an approximate algorithm. All BnB-related statistics are set to 0 and
    /// the solution is not optimal by definition.
    pub fn approx(
        value: T,
        approx_factor: f64,
        num_resources: usize,
        num_tasks: usize,
        elapsed: Duration,
    ) -> Self {
        Self {
            value,
            approx_factor,
            num_resources,
            num_tasks,
            elapsed,
            expanded: 0,
            pruned_value: 0,
            pruned_h_value: 0,
            pruned_closed: 0,
            proved_optimal: false,
        }
    }

    /// For an approximate algorithm returns an inclusive range of where the optimal schedule value
    /// is, otherwise `None` is returned (whenever `approx_factor.is_nan()`).
    ///
    /// # Example
    /// ```rust
    /// # extern crate makespan;
    /// use std::time::Duration;
    /// use makespan::Stats;
    ///
    /// let stats = Stats::approx(11., 1.4444444444444444, 3, 7, Duration::from_secs(1));
    ///
    /// let range = stats.optimality_range().unwrap();
    /// assert!(range.contains(&9.));
    /// ```
    pub fn optimality_range(&self) -> Option<RangeInclusive<T>> {
        if self.approx_factor.is_nan() {
            return None;
        }
        let start = self.value / T::from(self.approx_factor)?;
        Some(start..=self.value)
    }
}

/// Find `(argmin_x f(x), min_x f(x))` of function `f` on set `xs` or `None` if `xs` is empty.
/// If `xs` is a multi-set and the minimum is not strict, item with the lowest index is returned.
pub(crate) fn minimize<T, F>(xs: &[T], f: F) -> Option<(usize, T)>
where
    T: PartialOrd + Copy + Clone,
    F: Fn(T) -> T,
{
    if xs.is_empty() {
        return None;
    }

    let mut arg_min = 0;
    let mut min = f(xs[arg_min]);

    for (i, &x) in xs.iter().enumerate() {
        let fx = f(x);
        if fx < min {
            arg_min = i;
            min = fx;
        }
    }
    Some((arg_min, min))
}

/// Copy `processing_times` to a vector of pairs `(i, pt)` where `i` is the sequential index and
/// `pt` is original processing time wrapped into [OrderedFloat](struct.OrderedFloat.html).
#[inline]
pub(crate) fn preprocess<T: Float>(processing_times: &[T]) -> Vec<Task<T>> {
    processing_times
        .iter()
        .enumerate()
        .map(|(id, &pt)| Task { id, pt: pt.into() })
        .collect()
}

/// Sum given slice of `OrderedFloat`s.
#[inline]
pub(crate) fn sum<T: Float + Sum>(xs: &[OrderedFloat<T>]) -> OrderedFloat<T> {
    xs.iter().map(|t| t.into_inner()).sum::<T>().into()
}
