use std::collections::hash_map::{DefaultHasher, Entry};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::{Duration, SystemTime};

use num::Float;
use ordered_float::OrderedFloat;
use std::cmp::Ordering;

#[inline]
fn minimize<T, F>(xs: &[OrderedFloat<T>], f: F) -> Option<(usize, OrderedFloat<T>)>
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

#[derive(Debug)]
pub struct Solution<T>
where
    T: Float + Default,
{
    /// assignment: task j -> resource i (i.e. `schedule[j] = i`)
    pub schedule: Vec<usize>,
    /// total makespan (max completion time over all resources)
    pub value: T,
    /// approximation factor
    pub approx_factor: f64,
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
    ///     schedule: vec![0, 1, 0, 2], value: 2., approx_factor: 0., num_resources: 3,
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
    ///     schedule: vec![0, 1, 0, 2], value: 2., approx_factor: 0., num_resources: 3,
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

// TODO: split Solution and return this instead
//  - possibly include separate variants for BnB and approx.
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
    elapsed: Duration,
}

// TODO: maybe use as a return type for the schedule?
//  - however, API might not be that nice because once would have to check the instance even though
//  - just one would be expected
// pub enum Solution {
//     TASKS(Vec<usize>),
//     GANTT(HashMap<usize, Vec<usize>>),
// }

pub enum Scheduler {
    LPT,
    BnB { timeout: Option<Duration> },
}

// TODO: Allow T to be int type / any type that can be converted to f64
/// Solver for `P || C_max` task scheduling problem where
///   1. `P` stands for **parallel identical resources**
///   2. `||` denotes that there are no task constraints, notably that **tasks are non-preemptive**
///   3. `C_max` means that the objective is to **minimize the maximum completion time** (makespan)
///
/// This task is in general NP-hard and is sometimes called *Load balancing problem* or
/// *Minimum makespan scheduling*.
///
/// # Scheduler instances
/// There are two types of scheduler variants:
///  1. Approximate solver - `LPT`
///  2. Optimal solver - `BnB`
///
/// ## Approximate solver
/// LPT stands for *Longest Processing Time First* and is an approximation algorithm that runs in
/// `O(n*log(n))` time where `n` is the number of non-preemptive tasks to schedule.
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/kR` where
///  * `R` is no. resources
///  * `k` is no. tasks assigned to the resource which finishes last
///
/// ### Examples
///
/// Example where a sub-optimal solution is found.
/// ```
/// # extern crate makespan;
/// use makespan::Scheduler;
/// let pts = vec![5., 5., 4., 4., 3., 3., 3.];
/// let solution = Scheduler::LPT.schedule(&pts, 3).unwrap();
/// assert_eq!(solution.value, 11.); // opt = 9.
/// ```
///
/// ## Optimal solver
/// TODO
///
impl Scheduler {
    /// Run scheduling algorithm determined by this instance for given task `processing_times` and
    /// number of available resources (`num_resources`).
    ///
    /// The solution will generally exist except some edge cases when there are either no tasks or
    /// resources.
    pub fn schedule<T>(&self, processing_times: &[T], num_resources: usize) -> Option<Solution<T>>
    where
        T: Float + Default,
    {
        match self {
            Self::LPT => lpt(processing_times, num_resources),
            Self::BnB { timeout } => bnb(
                processing_times,
                num_resources,
                timeout.unwrap_or(Duration::from_secs_f64(Float::infinity())),
            ),
        }
    }
}

#[inline]
fn preprocess<T>(processing_times: &[T]) -> Vec<(usize, OrderedFloat<T>)>
where
    T: Float,
{
    processing_times
        .iter()
        .enumerate()
        .map(|(x, pt)| (x, OrderedFloat(*pt)))
        .collect::<Vec<(usize, OrderedFloat<T>)>>()
}

// Sort given slice of `(task, time)` pairs in non-increasing order of times.
#[inline]
fn sort_by_processing_time<T>(processing_times: &mut [(usize, OrderedFloat<T>)])
where
    T: Float,
{
    processing_times.sort_by(|(_, x), (_, y)| x.cmp(y).reverse());
}

fn lpt<T>(processing_times: &[T], num_resources: usize) -> Option<Solution<T>>
where
    T: Float + Default,
{
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    // collect pairs (task, processing time) to preserve original tasks - O(n)
    let mut processing_times = preprocess(processing_times);

    into_lpt_schedule(&mut processing_times, num_resources)
}

// Consume `processing_times` and compute LPT schedule (see `lpt`).
fn into_lpt_schedule<T>(
    processing_times: &mut [(usize, OrderedFloat<T>)],
    num_resources: usize,
) -> Option<Solution<T>>
where
    T: Float + Default,
{
    // sort tasks in non-increasing processing times - O(n * log(n))
    sort_by_processing_time(processing_times);

    lpt_sorted(processing_times, num_resources)
}

// Run LPT with the assumption that `processing_times` are given as pairs `(task, time)` and are
// already sorted by processing times.
fn lpt_sorted<T>(
    processing_times: &[(usize, OrderedFloat<T>)],
    num_resources: usize,
) -> Option<Solution<T>>
where
    T: Float + Default,
{
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let num_tasks = processing_times.len();

    let mut completion_times: Vec<OrderedFloat<T>> = vec![OrderedFloat::default(); num_resources];
    let mut task_dist = vec![0u32; num_resources];
    let mut schedule = vec![0usize; num_tasks];

    // greedily schedule each task to a resource that will have the smallest completion time - O(n)
    for (task, pt) in processing_times.iter() {
        let min = minimize(&completion_times, |ct| OrderedFloat(*ct + **pt));
        if let Some((min_resource, min_completion)) = min {
            completion_times[min_resource] = min_completion;
            task_dist[min_resource] += 1;
            schedule[*task] = min_resource;
        }
    }

    // compute final objective value (C_max) - O(R)
    let value = completion_times
        .into_iter()
        .max()
        .unwrap_or(OrderedFloat::default());
    // let value = completion_times.into_iter().max_by(cmp_f64).unwrap_or(0f64);

    // compute approximation factor r(LPT) - O(R)
    let approx_factor = task_dist
        .into_iter()
        .max()
        .map(|m| 1. + (1. / m as f64) + (1. / (m * num_resources as u32) as f64))
        .unwrap_or(1f64);

    Some(Solution {
        schedule,
        value: value.0,
        approx_factor,
        num_resources,
    })
}

struct BnBState<T>
where
    T: Float + Default,
{
    id: u64,
    schedule: Vec<usize>,
    completion_times: Vec<OrderedFloat<T>>,
    value: OrderedFloat<T>,
    h_value: OrderedFloat<T>,
}

impl<T> BnBState<T>
where
    T: Float + Default,
{
    fn compute_hash(completion_times: &[OrderedFloat<T>]) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut completion_times = completion_times.to_vec();
        completion_times.sort();
        completion_times.hash(&mut hasher);
        hasher.finish()
    }

    fn init(num_resources: usize, h_value: T) -> Self {
        let completion_times = vec![OrderedFloat::default(); num_resources];
        BnBState {
            id: Self::compute_hash(&completion_times),
            schedule: Vec::new(),
            completion_times,
            value: OrderedFloat::default(),
            h_value: OrderedFloat(h_value),
        }
    }
}

impl<T> Hash for BnBState<T>
where
    T: Float + Default,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }

    fn hash_slice<H: Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        for s in data.iter() {
            s.id.hash(state);
        }
    }
}

impl<T> PartialEq for BnBState<T>
where
    T: Float + Default,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn ne(&self, other: &Self) -> bool {
        !self.eq(other)
    }
}

impl<T> Eq for BnBState<T> where T: Float + Default {}

impl<T> PartialOrd for BnBState<T>
where
    T: Float + Default,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for BnBState<T>
where
    T: Float + Default,
{
    fn cmp(&self, other: &BnBState<T>) -> Ordering {
        // reverse because BnB uses a max-heap and our objective is to minimize
        self.h_value.cmp(&other.h_value).reverse()
    }
}

fn bnb<T>(processing_times: &[T], num_resources: usize, timeout: Duration) -> Option<Solution<T>>
where
    T: Float + Default,
{
    let start = SystemTime::now();

    // TODO: deduplicate
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let num_tasks = processing_times.len();

    let mut processing_times = preprocess(processing_times);

    let mut expanded = 0u32;
    let mut pruned_value = 0u32;
    let mut pruned_h_value = 0u32;
    let mut pruned_closed = 0u32;
    let mut proved_optimal = true;

    // pre-sorted set of tasks:
    //  1. LPT does it anyway
    //  2. Heuristic: larger tasks might prude the space sooner
    sort_by_processing_time(&mut processing_times);

    // closed set will store just state hashes (ids) as unique identifiers to save memory
    let mut closed: HashSet<u64> = HashSet::new();

    // run LPT to get lower bound (LB) on the solution
    // TODO: should be BnBState
    let mut best = lpt_sorted(&processing_times, num_resources)?;

    let mut queue = BinaryHeap::new();
    queue.push(BnBState::init(num_resources, best.value));

    while let Some(node) = queue.pop() {
        // return current best if running for more than given time limit
        let elapsed = start.elapsed().expect("Failed to get elapsed system time.");
        if elapsed > timeout {
            proved_optimal = false;
            break;
        }

        // TODO: implement
    }

    None
}

#[cfg(test)]
mod tests {
    use crate::{Scheduler, Solution};
    use std::collections::HashMap;

    #[test]
    fn solution_gantt_schedule() {
        let solution = Solution {
            schedule: vec![0, 1, 0, 2],
            value: 2.,
            approx_factor: 0.,
            num_resources: 3,
        };

        let mut expected = HashMap::new();
        expected.insert(0, vec![0, 2]);
        expected.insert(1, vec![1]);
        expected.insert(2, vec![3]);

        assert_eq!(solution.gantt_schedule(), expected);
    }

    #[test]
    fn lpt_trivial_cases() {
        let scheduler = Scheduler::LPT;

        // no tasks
        let pts: Vec<f64> = Vec::new();
        assert!(scheduler.schedule(&pts, 2).is_none());

        // no resources
        let pts = vec![10., 20.];
        assert!(scheduler.schedule(&pts, 0).is_none());

        // single resource
        let solution = scheduler.schedule(&pts, 1).unwrap();
        assert_eq!(solution.schedule, vec![0; pts.len()]);
        assert_eq!(solution.value, 30.);
        assert_eq!(solution.approx_factor, 2.);
    }

    #[test]
    fn non_trivial_lpt_schedule() {
        let pts = vec![5., 5., 4., 4., 3., 3., 3.];
        let solution = Scheduler::LPT.schedule(&pts, 3).unwrap();

        // schedule (LPT with stable sort is deterministic)
        assert_eq!(solution.schedule, vec![0, 1, 2, 2, 0, 1, 0]);

        // schedule contains all resources from input range
        let mut resources = solution.schedule.to_vec();
        resources.sort();
        resources.dedup();
        assert_eq!(resources, vec![0, 1, 2]);

        // task distribution is correct
        assert_eq!(solution.task_loads(), vec![3, 2, 2]);

        // objective value is correct
        assert_eq!(solution.value, 11.); // opt: 9.

        // approx. factor is correct
        assert_eq!(solution.approx_factor, 1.4444444444444444);
    }
}
