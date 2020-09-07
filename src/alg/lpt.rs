use std::time::SystemTime;

use num::Float;
use ordered_float::OrderedFloat;

use crate::alg::core::*;

/// Search for approx. solution of `P || C_max` using LPT (Longest Processing Time First) algorithm.
///
/// Asymptotic runtime is `O(n*log(n))` time where `n` is the number of non-preemptive tasks.
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/kR` where
///  * `R` is no. resources
///  * `k` is no. tasks assigned to the resource which finishes last
pub(crate) fn lpt<T>(
    processing_times: &[T],
    num_resources: usize,
) -> Option<(Solution<T>, Stats<T>)>
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

/// Consume `processing_times` and compute LPT schedule (see `lpt`).
pub(crate) fn into_lpt_schedule<T>(
    processing_times: &mut [(usize, OrderedFloat<T>)],
    num_resources: usize,
) -> Option<(Solution<T>, Stats<T>)>
where
    T: Float + Default,
{
    // sort tasks in non-increasing processing times - O(n * log(n))
    sort_by_processing_time(processing_times);

    lpt_sorted(processing_times, num_resources)
}

/// Run LPT with the assumption that `processing_times` are given as pairs `(task, time)` and are
/// already sorted by processing times.
pub(crate) fn lpt_sorted<T>(
    processing_times: &[(usize, OrderedFloat<T>)],
    num_resources: usize,
) -> Option<(Solution<T>, Stats<T>)>
where
    T: Float + Default,
{
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let start = SystemTime::now();

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
