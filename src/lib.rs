use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::HashMap;

#[inline]
fn cmp_f64(x: &f64, y: &f64) -> Ordering {
    if x < y {
        Ordering::Less
    } else if x > y {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

#[inline]
fn minimize<F>(xs: &[f64], f: F) -> Option<(usize, f64)>
where
    F: Fn(f64) -> f64,
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
pub struct Solution {
    /// assignment: task j -> resource i (i.e. `schedule[j] = i`)
    pub schedule: Vec<usize>,
    /// total makespan (max completion time over all resources)
    pub value: f64,
    /// approximation factor
    pub approx_factor: f64,
    /// number of resources
    pub num_resources: usize,
}

impl Solution {
    /// Generate schedule in the form of a Gantt chart, i.e. as mapping: `resource -> [tasks]`.
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

// TODO: Generalize from &[f64] to generic numeric type T: Add + Ord
//  - must deal with the fact that f64 is not Ord

/// LPT (Longest Processing Time First) approximation algorithm for `P || C_max`
/// (NP-hard in general).
///
/// Runs in `O(n*log(n))` time where `n` is the number of non-preemptive tasks to schedule
/// (i.e. `n = |processing_times|`).
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/kR` where
///  * `R = num_resources`
///  * `k` is no. tasks assigned to the resource which finishes last
///
/// Example where a sub-optimal solution is found.
/// ```
/// # extern crate makespan;
/// use makespan;
/// let pts = vec![5., 5., 4., 4., 3., 3., 3.];
/// let solution = makespan::lpt(&pts, 3).unwrap();
/// assert_eq!(solution.value, 11.); // opt: 9.
/// ```
pub fn lpt(processing_times: &[f64], num_resources: usize) -> Option<Solution> {
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    // collect pairs (task, processing time) to preserve original tasks - O(n)
    let mut processing_times = processing_times
        .iter()
        .enumerate()
        .map(|(x, pt)| (x, *pt))
        .collect::<Vec<(usize, f64)>>();

    into_lpt_schedule(&mut processing_times, num_resources)
}

/// Consume `processing_times` and compute LPT schedule (see `lpt`).
pub fn into_lpt_schedule(
    processing_times: &mut [(usize, f64)],
    num_resources: usize,
) -> Option<Solution> {
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let num_tasks = processing_times.len();

    // sort tasks in non-increasing processing times - O(n * log(n))
    processing_times.sort_by(|(_, x), (_, y)| cmp_f64(x, y).reverse());

    let mut completion_times = vec![0f64; num_resources];
    let mut task_dist = vec![0u32; num_resources];
    let mut schedule = vec![0usize; num_tasks];

    // greedily schedule each task to a resource that will have the smallest completion time - O(n)
    for (ref task, ref pt) in processing_times.into_iter() {
        let min = minimize(&completion_times, |ct| ct + *pt);
        if let Some((min_resource, min_completion)) = min {
            completion_times[min_resource] = min_completion;
            task_dist[min_resource] += 1;
            schedule[*task] = min_resource;
        }
    }

    // compute final objective value (C_max) - O(R)
    let value = completion_times.into_iter().max_by(cmp_f64).unwrap_or(0f64);

    // compute approximation factor r(LPT) - O(R)
    let approx_factor = task_dist
        .into_iter()
        .max()
        .map(|m| 1. + (1. / m as f64) + (1. / (m * num_resources as u32) as f64))
        .unwrap_or(1f64);

    Some(Solution {
        schedule,
        value,
        approx_factor,
        num_resources,
    })
}

#[cfg(test)]
mod tests {
    use crate::{cmp_f64, lpt, minimize, Solution};
    use std::cmp::Ordering;
    use std::collections::HashMap;

    #[test]
    fn simple_cmp_f64_tests() {
        assert_eq!(cmp_f64(&1.2, &-2.3), Ordering::Greater);
        assert_eq!(cmp_f64(&1.2, &3.4), Ordering::Less);
        assert_eq!(cmp_f64(&1.2, &1.2), Ordering::Equal);
    }

    #[test]
    fn minimize_empty_list() {
        assert_eq!(minimize(&Vec::new(), |x| x), None);
    }

    #[test]
    fn minimize_simple_list_with_identity() {
        let xs = vec![0., -2., 3., 1.];
        assert_eq!(minimize(&xs, |x| x), Some((1, -2.)));
    }

    #[test]
    fn minimize_simple_list_with_transformation() {
        let xs = vec![0., -2., 3., 1.];
        assert_eq!(minimize(&xs, |x| x + 1.), Some((1, -1.)));
    }

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
        // no tasks
        let pts: Vec<f64> = Vec::new();
        assert!(lpt(&pts, 2).is_none());

        // no resources
        let pts = vec![10., 20.];
        assert!(lpt(&pts, 0).is_none());

        // single resource
        let solution = lpt(&pts, 1).unwrap();
        assert_eq!(solution.schedule, vec![0; pts.len()]);
        assert_eq!(solution.value, 30.);
        assert_eq!(solution.approx_factor, 2.);
    }

    #[test]
    fn non_trivial_lpt_schedule() {
        let pts = vec![5., 5., 4., 4., 3., 3., 3.];
        let solution = lpt(&pts, 3).unwrap();

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
