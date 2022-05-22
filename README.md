![makespan](https://github.com/matyama/makespan/workflows/makespan/badge.svg)
![Maintenance](https://img.shields.io/badge/maintenance-experimental-blue.svg)

# makespan

## Makespan
This Rust library implements several solvers for task scheduling problems optimizing
[makespan](https://en.wikipedia.org/wiki/Makespan).

### Graham's notation
Definitions of scheduling problems in *operations research* commonly use
[Graham's notation](https://en.wikipedia.org/wiki/Optimal_job_scheduling) `a|b|c` where
  1. `a` defines *machine environment* (resources)
  2. `b` denotes *task/job characteristics*
  3. `c` specifies the *objective function* that is to be minimized (note that this library
     focuses on optimizing the *makespan* `C_max`)

## Covered scheduling problems
The module structure of this library follows a structure of the general taxonomy of scheduling
problems and as mentioned above focuses on minimizing the *makespan* (`C_max`).

General constraints:
 - Each task can be processed by at most one resource at a time
 - Each resource can process at most one task at a time

### Single-processor
This class of scheduling problems considers single processing unit (resource).

#### Non-preemptive
This sub-class considers non-preemptive tasks and is covered by module [`sp`](crate::sp). The
list of problems and corresponding algorithms includes:
 - `1||C_max`: optimal, linear
 - `1|prec|C_max`: optimal, linear
 - `1|r_j|C_max`: optimal, log-linear
 - `1|d'_j|C_max`: EDF (optimal, log-linear)
 - `1|r_j,d'_j|C_max`: heuristic (log-linear), Bratley's BnB (optimal)
 - `1|chains,r_j|C_max`: special case with `chains` totally ordered (otpimal, linear)

### Multi-processor
This class of scheduling problems considers multiple processing units (resources).

#### Non-preemptive
This sub-class considers non-preemptive tasks and is covered by module [`mp`](crate::mp). The
list of problems and corresponding algorithms includes:
 - `P||C_max`: LPT (approx), BnB (optimal)

#### Preemptive
This sub-class considers preemptive tasks and is covered by module [`mp_pmtn`](crate::mp_pmtn).
The list of problems and corresponding algorithms includes:
 - `P|pmtn|C_max`: McNaughton (optimal)

## Resources
 - [Complexity results](http://www2.informatik.uni-osnabrueck.de/knust/class/dateien/allResults.pdf)
 - [The scheduling zoo](http://www-desir.lip6.fr/~durrc/query/)
 - [CTU lecture slides](https://rtime.ciirc.cvut.cz/~hanzalek/KO/sched_e.pdf)

## License and version
**Current version**: 0.3.0

**License**: MIT OR Apache-2.0
