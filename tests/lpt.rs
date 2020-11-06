use makespan::Scheduler;

const SCHEDULER: Scheduler = Scheduler::LPT;

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

    // task distribution is correct (except symmetries, hence the sort)
    let mut dist = solution.task_loads();
    dist.sort();
    assert_eq!(dist, vec![2, 2, 3]);

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
fn test_subopt_instance() {
    fn subopt_instance(num_resources: usize) -> Vec<f64> {
        let mut pts = vec![0usize; 2 * num_resources + 1];
        let mut pt = 2 * num_resources - 1;
        while pt >= num_resources {
            pts.push(pt);
            pts.push(pt);
            pt -= 1;
        }
        pts.push(num_resources);
        pts.into_iter().map(|pt| pt as f64).collect()
    }

    let res = 100;
    let pts = subopt_instance(res);
    let _ = SCHEDULER.schedule(&pts, res).unwrap();
}
