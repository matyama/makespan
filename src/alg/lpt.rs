use std::time::SystemTime;

use num_traits::Float;
use ordered_float::OrderedFloat;
use voracious_radix_sort::RadixSort;

use crate::alg::core::Task;
use crate::alg::core::*;

/// Search for approx. solution of `P || C_max` using LPT (Longest Processing Time First) algorithm.
///
/// Asymptotic runtime is `O(n)` time where `n` is the number of non-preemptive tasks (note: this
/// implementation uses a *RadixSort* to sort tasks in linear time).
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/kR` where
///  * `R` is no. resources (`R << n`)
///  * `k` is no. tasks assigned to the resource which finishes last
pub(crate) fn lpt<T>(
    processing_times: &[T],
    num_resources: usize,
) -> Option<(Solution<T>, Stats<T>)>
where
    T: Float + Default + Send + Sync,
{
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let start = SystemTime::now();

    // collect tasks {id, processing time} to preserve original order - O(n)
    let mut tasks = preprocess(processing_times);

    // sort tasks in non-increasing processing times - O(n)
    tasks.voracious_sort();

    // Assign sorted tasks in greedy way - O(n)
    greedy_schedule(&tasks, num_resources, start)
}

/// Assume that `tasks` are given as instances of `Task` and are
/// already sorted by processing times. Then this function sequentially assigns tasks to resources
/// while minimizing maximum completion time (makespan).
pub(crate) fn greedy_schedule<T>(
    tasks: &[Task<T>],
    num_resources: usize,
    start: SystemTime,
) -> Option<(Solution<T>, Stats<T>)>
where
    T: Float + Default + Send + Sync,
{
    if tasks.is_empty() || num_resources == 0 {
        return None;
    }

    let num_tasks = tasks.len();

    let mut completion_times: Vec<OrderedFloat<T>> = vec![OrderedFloat::default(); num_resources];
    let mut task_dist = vec![0u32; num_resources];
    let mut schedule = vec![0usize; num_tasks];

    // greedily schedule each task to a resource that will have the smallest completion time - O(n)
    for task in tasks.iter() {
        let min = minimize(&completion_times, |ct| ct + task.pt);
        if let Some((min_resource, min_completion)) = min {
            completion_times[min_resource] = min_completion;
            task_dist[min_resource] += 1;
            schedule[task.id] = min_resource;
        }
    }

    // compute final objective value (C_max) - O(R)
    let value = completion_times.into_iter().max().unwrap_or_default();

    // compute approximation factor r(LPT) - O(R)
    let approx_factor = task_dist
        .into_iter()
        .max()
        .map(|m| 1. + (1. / m as f64) + (1. / (m * num_resources as u32) as f64))
        .unwrap_or(1.);

    let solution = Solution {
        schedule,
        value: value.0,
        num_resources,
    };

    let elapsed = start.elapsed().expect("Failed to get elapsed system time.");
    let stats = Stats::approx(
        value.into_inner(),
        approx_factor,
        num_resources,
        num_tasks,
        elapsed,
    );

    Some((solution, stats))
}
