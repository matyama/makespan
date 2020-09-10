use makespan::Solution;
use std::collections::HashMap;

#[test]
fn gantt_schedule() {
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
fn task_distribution_to_resources() {
    // no tasks
    let solution = makespan::Solution {
        schedule: vec![],
        value: 0.,
        num_resources: 3,
    };
    assert_eq!(solution.task_loads(), vec![0, 0, 0]);

    // single resource
    let solution = makespan::Solution {
        schedule: vec![0; 4],
        value: 4.,
        num_resources: 1,
    };
    assert_eq!(solution.task_loads(), vec![4]);

    // non-trivial distribution
    let solution = makespan::Solution {
        schedule: vec![0, 1, 0, 2],
        value: 2.,
        num_resources: 3,
    };
    assert_eq!(solution.task_loads(), vec![2, 1, 1]);
}
