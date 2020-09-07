use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::time::Duration;

use num::Float;
use ordered_float::OrderedFloat;

// TODO: ideally impl something like Add for OrderedFloat<T>

/// Data structure holding resulting `schedule` and objective `value`.
///
/// The schedule is represented as a vector of resources which is indexed by tasks. Additionally,
/// this structure records the value of the schedule (maximum completion time) and original number
/// of resources.
#[derive(Debug)]
pub struct Solution<T>
where
    T: Float + Default,
{
    /// assignment: task j -> resource i (i.e. `schedule[j] = i`)
    pub schedule: Vec<usize>,
    /// total makespan (max completion time over all resources)
    pub value: T,
    /// number of resources
    pub num_resources: usize,
}

impl<T> Solution<T>
where
    T: Float + Default,
{
    /// Generate schedule in the form of a Gantt chart, i.e. as mapping: `resource -> [tasks]`.
    ///
    /// # Example
    /// ```
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

        for (task, resource) in self.schedule.iter().enumerate() {
            match gantt.entry(*resource) {
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
    /// ```
    /// # extern crate makespan;
    /// let solution = makespan::Solution {
    ///     schedule: vec![0, 1, 0, 2], value: 2., num_resources: 3,
    /// };
    /// assert_eq!(solution.task_loads(), vec![2, 1, 1]);
    /// ```
    pub fn task_loads(&self) -> Vec<u32> {
        let mut task_dist = vec![0u32; self.num_resources];
        for resource in self.schedule.iter() {
            task_dist[*resource] += 1;
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
pub struct Stats<T>
where
    T: Float + Default,
{
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

impl<T> Stats<T>
where
    T: Float + Default,
{
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
}

/// Find `(argmin_x f(x), min_x f(x))` of function `f` on set `xs` or `None` if `xs` is empty.
/// If `xs` is a multi-set and the minimum is not strict, item with the lowest index is returned.
#[inline]
pub(crate) fn minimize<T, F>(xs: &[OrderedFloat<T>], f: F) -> Option<(usize, OrderedFloat<T>)>
where
    T: Float,
    F: Fn(OrderedFloat<T>) -> OrderedFloat<T>,
{
    if xs.is_empty() {
        return None;
    }

    let mut arg_min = 0;
    let mut min = f(xs[arg_min]);

    for (i, x) in xs.iter().enumerate() {
        let fx = f(*x);
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
pub(crate) fn preprocess<T>(processing_times: &[T]) -> Vec<(usize, OrderedFloat<T>)>
where
    T: Float,
{
    processing_times
        .iter()
        .enumerate()
        .map(|(x, pt)| (x, OrderedFloat(*pt)))
        .collect::<Vec<(usize, OrderedFloat<T>)>>()
}

/// Sort given slice of `(task, time)` pairs in place in non-increasing order of times.
#[inline]
pub(crate) fn sort_by_processing_time<T>(processing_times: &mut [(usize, OrderedFloat<T>)])
where
    T: Float,
{
    processing_times.sort_by(|(_, x), (_, y)| x.cmp(y).reverse());
}
