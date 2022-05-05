//! # Task scheduler
//! This library implements several solvers for `P || C_max` task scheduling problem.
//!
//! `P || C_max` is a `a|b|c` notation used in *operations research* where
//!   1. `P` stands for **parallel identical resources**
//!   2. `||` denotes that there are no task constraints, notably that **tasks are non-preemptive**
//!   3. `C_max` means that the objective is to **minimize the maximum completion time** (makespan)
//!
//! This task is in general NP-hard and is sometimes called *Load balancing problem* or
//! *Minimum makespan scheduling*.
//!
//! ## Scheduler variants
//! There are two algorithms for solving the minimum makespan problem:
//!  1. Approximate solver - `LPT` (Longest Processing Time First)
//!  2. Optimal solver - `BnB` (Branch & Bound Best-first search)
//!
//! ## Example
//! ```rust
//! # extern crate makespan;
//! use makespan::Scheduler;
//!
//! // Define a vector of processing times for each task
//! let processing_times = vec![5., 5., 4., 4., 3., 3., 3.];
//!
//! // Create new Branch & Bound scheduler
//! let scheduler = Scheduler::BnB { timeout: None };
//!
//! // Find optimal schedule for 3 resources
//! let (solution, stats) = scheduler.schedule(&processing_times, 3).unwrap();
//!
//! for (task, resource) in solution.schedule.iter().enumerate() {
//!     println!("{} <- {}", resource, task);
//! }
//!
//! assert_eq!(solution.value, 9.);
//! ```
//!
//! # Related PoC project
//! There is a related project called [makespan-poc](https://github.com/matyama/makespan-poc) which
//! contains several Jupyter notebooks exporing scheduling algorithms and their potential.
//!
//! Contributors are welcome to implement and test new algorithms in there first. The PoC repo is
//! meant for *quick-and-dirty* prototyping and general concept validation.

use std::iter::Sum;
use std::time::Duration;

use num_traits::Float;

mod alg;

use alg::{BranchAndBound, LongestProcessingTimeFirst, Solve};
pub use alg::{Solution, Stats};

// TODO: Is this enum really necessary now that there is `Solve` and structs impl. it?
// TODO: Allow T to be int type / any type that can be converted to f64
/// Solver for `P || C_max` task scheduling problem. The setting is
///  * parallel identical resources
///  * no task constraints
///  * minimization of maximum task completion time
///
/// # Scheduler instances
/// There are two scheduler variants:
///  1. Approximate solver - `Lpt`
///  2. Optimal solver - `BnB`
///
/// ## Approximate solver
/// LPT stands for *Longest Processing Time First* and is an approximation algorithm that runs in
/// `O(n*log(n))` time where `n` is the number of non-preemptive tasks to schedule.
///
/// Approximation factor is `r(LPT) = 1 + 1/k − 1/kR` where
///  * `R` is no. resources (generally assumbed to be `R << n`)
///  * `k` is no. tasks assigned to the resource which finishes last
///
/// ### Examples
///
/// Example where a sub-optimal solution is found.
/// ```rust
/// # extern crate makespan;
/// use makespan::Scheduler;
/// let pts = vec![5., 5., 4., 4., 3., 3., 3.];
/// let (solution, stats) = Scheduler::Lpt.schedule(&pts, 3).unwrap();
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
/// `R` is the number of resources and `n` is again the number of tasks (`R << n`).
///
/// As mentioned above, this algorithm uses a Best-first search under the hood which means that
/// nodes to expand in the BnB tree are picked based on a heuristic function `h(N)`. The heuristic
/// is a simple *admissible* estimate of the value of partial solution `N` obtained by scheduling
/// remaining tasks as if the problem was `P |pmtn| C_max` - i.e. relaxes the problem by allowing
/// task preemption. Moreover, in current implementation we allow just one interrupt per task and
/// these two task splits cannot run at the same time.
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
/// ```rust
/// # extern crate makespan;
/// use makespan::Scheduler;
/// let pts = vec![5., 5., 4., 4., 3., 3., 3.];
/// let (solution, stats) = Scheduler::BnB { timeout: None }.schedule(&pts, 3).unwrap();
/// assert_eq!(solution.value, 9.);
/// assert!(stats.proved_optimal);
/// ```
pub enum Scheduler {
    /// Longest Processing Time First approximate solver
    Lpt,
    /// Branch & Bound optimal solver
    BnB { timeout: Option<Duration> },
}

// TODO: type alias T or better add custom one to type check that processing time is positive
impl Scheduler {
    /// Run scheduling algorithm determined by this instance for given task `processing_times` and
    /// number of available resources (`num_resources`).
    ///
    /// The solution will generally exist except for some edge cases when there are either no tasks
    /// or resources.
    pub fn schedule<T: Float + Sum>(
        &self,
        processing_times: &[T],
        num_resources: usize,
    ) -> Option<(Solution<T>, Stats<T>)> {
        match self {
            Self::Lpt => LongestProcessingTimeFirst.solve(processing_times, num_resources),
            Self::BnB { timeout } => {
                BranchAndBound { timeout: *timeout }.solve(processing_times, num_resources)
            }
        }
    }
}
