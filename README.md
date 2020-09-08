![makespan](https://github.com/matyama/makespan/workflows/makespan/badge.svg)
![Maintenance](https://img.shields.io/badge/maintenance-activly--developed-brightgreen.svg)

# makespan

## Task scheduler
This library implements several solvers for `P || C_max` task scheduling problem.

`P || C_max` is a `a|b|c` notation used in *operations research* where
  1. `P` stands for **parallel identical resources**
  2. `||` denotes that there are no task constraints, notably that **tasks are non-preemptive**
  3. `C_max` means that the objective is to **minimize the maximum completion time** (makespan)

This task is in general NP-hard and is sometimes called *Load balancing problem* or
*Minimum makespan scheduling*.

### Scheduler variants
There are two algorithms for solving the minimum makespan problem:
 1. Approximate solver - `LPT` (Longest Processing Time First)
 2. Optimal solver - `BnB` (Branch & Bound Best-first search)

### Example
```rust
use makespan::Scheduler;

// Define a vector of processing times for each task
let processing_times = vec![5., 5., 4., 4., 3., 3., 3.];

// Create new Branch & Bound scheduler
let scheduler = Scheduler::BnB { timeout: None };

// Find optimal schedule for 3 resources
let (solution, stats) = scheduler.schedule(&processing_times, 3).unwrap();

for (task, resource) in solution.schedule.iter().enumerate() {
    println!("{} <- {}", resource, task);
}

assert_eq!(solution.value, 9.);
```

**Current version**: 0.1.0

**License**: MIT OR Apache-2.0
