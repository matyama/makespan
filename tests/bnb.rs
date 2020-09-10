use makespan::Scheduler;

const SCHEDULER: Scheduler = Scheduler::BnB { timeout: None };

#[test]
fn no_tasks() {
    let pts: Vec<f64> = Vec::new();
    assert!(SCHEDULER.schedule(&pts, 2).is_none());
}

#[test]
fn no_resources() {
    let pts = vec![10., 20.];
    assert!(SCHEDULER.schedule(&pts, 0).is_none());
}

#[test]
fn single_resource() {
    let pts = vec![10., 20.];
    let (solution, _) = SCHEDULER.schedule(&pts, 1).unwrap();
    assert_eq!(solution.schedule, vec![0; pts.len()]);
    assert_eq!(solution.value, 30.);
}

#[test]
fn non_trivial_schedule() {
    let pts = vec![5., 5., 4., 4., 3., 3., 3.];
    let (solution, stats) = SCHEDULER.schedule(&pts, 3).unwrap();

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
