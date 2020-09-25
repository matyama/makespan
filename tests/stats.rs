use makespan::Stats;
use std::time::Duration;

#[test]
fn some_optimality_range() {
    let stats = Stats::approx(11., 1.4444444444444444, 3, 7, Duration::from_secs(1));

    let range = stats.optimality_range();
    assert!(range.is_some());

    let range = range.unwrap();
    assert!(range.contains(&9.));

    assert_eq!(*range.start(), 7.615384615384616);
    assert_eq!(*range.end(), 11.);
}

#[test]
fn none_optimality_range() {
    let stats = Stats::approx(11., f64::NAN, 3, 7, Duration::from_secs(1));
    assert!(stats.optimality_range().is_none());
}
