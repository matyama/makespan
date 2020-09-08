use std::iter::Sum;
use std::time::Duration;

use num::Float;

mod alg;

pub use alg::{Solution, Stats};

// TODO: Allow T to be int type / any type that can be converted to f64
/// Solver for `P || C_max` task scheduling problem.
///
/// `P || C_max` is a `a|b|c` notation used in *operations research* where
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
/// let (solution, stats) = Scheduler::LPT.schedule(&pts, 3).unwrap();
/// assert_eq!(solution.value, 11.); // opt = 9.
/// assert_eq!(stats.approx_factor, 1.4444444444444444); // 11. < 9. * 1.4444444444444444 = 13.
/// ```
///
/// ## Optimal solver
/// Optimal solver impements Branch and Bound (BnB) Best-first search. This algorithm is complete
/// and optimal, i.e. finds optimal schedule given enough time to run (controlled by `timeout`).
///
/// Because the scheduling problem is known to be NP-hard, it is believed that there is no
/// polynomial algorithm. Then it is not surprising that this implementation runs in `O(R^n)` where
/// `R` is the number of resources and `n` is again the number of tasks.
///
/// As mentioned above, this algorithm uses a Best-first search under the hood which means that
/// nodes to expand in the BnB tree are picked based on a heuristic function `h(N)`. The heuristic
/// is a simple *admissible* estimate of the value of partial solution `N` obtained by scheduling
/// remaining tasks as if the problem was `P |pmtn| C_max` - i.e. relaxes the problem by allowing
/// task preemption. Moreover, in current implementation we allow indefinite number of pauses and
/// resumes and tasks can have inner parallelism (more parts can run at the same time).
///
/// Additional optimization techniques include:
///  * Initial best solution `B` is found by LPT and the search keeps only nodes with `f(N) < f(B)`
///  * Pruning based on `h(N)` (i.e. `h(N) < f(B)` must hold to expand node `N`)
///  * Pruning resource and task symmetries (e.g. with two identical tasks j1, j2 these assignments
///    are symmetric: `[0 <- j1, 1 <- j2] =sym= [0 <- j2, 1 <- j1]`)
///  * Tasks are pre-sorted by processing times (non-increasingly) and BnB then starts with longer
///    tasks first when it selects next task to extend the partial solution with. This heuristic
///    should in theory restrict the space more early in the search and thus help to prune
///    sub-optimal solutions.
///
/// ### Examples
///
/// Example where an optimal solution is better than the one found by LPT.
/// ```
/// # extern crate makespan;
/// use makespan::Scheduler;
/// let pts = vec![5., 5., 4., 4., 3., 3., 3.];
/// let (solution, stats) = Scheduler::BnB { timeout: None }.schedule(&pts, 3).unwrap();
/// assert_eq!(solution.value, 9.);
/// assert!(stats.proved_optimal);
/// ```
pub enum Scheduler {
    /// Longest Processing Time First approximate solver
    LPT,
    /// Branch & Bound optimal solver
    BnB { timeout: Option<Duration> },
}

impl Scheduler {
    /// Run scheduling algorithm determined by this instance for given task `processing_times` and
    /// number of available resources (`num_resources`).
    ///
    /// The solution will generally exist except some edge cases when there are either no tasks or
    /// resources.
    pub fn schedule<T>(
        &self,
        processing_times: &[T],
        num_resources: usize,
    ) -> Option<(Solution<T>, Stats<T>)>
    where
        T: Float + Default + Sum,
    {
        match self {
            Self::LPT => alg::lpt(processing_times, num_resources),
            Self::BnB { timeout } => alg::bnb(processing_times, num_resources, *timeout),
        }
    }
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
            num_resources: 3,
        };

        let mut expected = HashMap::new();
        expected.insert(0, vec![0, 2]);
        expected.insert(1, vec![1]);
        expected.insert(2, vec![3]);

        assert_eq!(solution.gantt_schedule(), expected);
    }

    #[test]
    fn trivial_cases() {
        let schedulers = [Scheduler::LPT, Scheduler::BnB { timeout: None }];

        for scheduler in schedulers.iter() {
            // no tasks
            let pts: Vec<f64> = Vec::new();
            assert!(scheduler.schedule(&pts, 2).is_none());

            // no resources
            let pts = vec![10., 20.];
            assert!(scheduler.schedule(&pts, 0).is_none());

            // single resource
            let (solution, _) = scheduler.schedule(&pts, 1).unwrap();
            assert_eq!(solution.schedule, vec![0; pts.len()]);
            assert_eq!(solution.value, 30.);
        }
    }

    #[test]
    fn non_trivial_lpt_schedule() {
        let pts = vec![5., 5., 4., 4., 3., 3., 3.];
        let (solution, stats) = Scheduler::LPT.schedule(&pts, 3).unwrap();

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
        assert_eq!(stats.value, solution.value);

        // approx. factor is correct
        assert_eq!(stats.approx_factor, 1.4444444444444444);

        assert!(!stats.proved_optimal);
    }

    #[test]
    fn non_trivial_bnb_schedule() {
        let scheduler = Scheduler::BnB { timeout: None };

        let pts = vec![5., 5., 4., 4., 3., 3., 3.];
        let (solution, stats) = scheduler.schedule(&pts, 3).unwrap();

        // schedule contains all resources from input range
        let mut resources = solution.schedule.to_vec();
        resources.sort();
        resources.dedup();
        assert_eq!(resources, vec![0, 1, 2]);

        // task distribution is correct (except symmetries, hence the sort)
        let mut dist = solution.task_loads();
        dist.sort();
        assert_eq!(dist, vec![2, 2, 3]);

        // objective value is correct and optimal
        assert_eq!(solution.value, 9.);
        assert_eq!(stats.value, solution.value);

        // approx. factor set to 1.
        assert_eq!(stats.approx_factor, 1.);

        // compared to LPT, there should be some expanded nodes from the search tree
        assert!(stats.expanded > 0);

        // optimality should be known
        assert!(stats.proved_optimal);
    }
}
