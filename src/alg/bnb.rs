use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::convert::identity;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::time::{Duration, SystemTime};

use num_traits::Float;
use ordered_float::OrderedFloat;

use crate::alg::core::*;
use crate::alg::lpt::greedy_schedule;

// TODO: generalize over heuristic
/// Search for optimal solution of `P || C_max` using Best-first BnB.
///
/// The *heuristic* `h(N)` of a node `N` (partial solution) is the minimum makespan of a schedule
/// that is initialized to the partial solution of `N` and completed by relaxing the problem and
/// allowing task preemption (i.e. remaining tasks are scheduled as `P |pmtn| C_max`).
pub(crate) fn bnb<T>(
    processing_times: &[T],
    num_resources: usize,
    timeout: Option<Duration>,
) -> Option<(Solution<T>, Stats<T>)>
where
    T: Float + Default + Sum,
{
    let start = SystemTime::now();

    // TODO: deduplicate
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let num_tasks = processing_times.len();

    let mut processing_times = preprocess(processing_times);

    // pre-sorted set of tasks:
    //  1. LPT does it anyway
    //  2. Heuristic: larger tasks might prude the space sooner
    sort_by_processing_time(&mut processing_times);

    let total_time = compute_total_time(&processing_times);

    // closed set will store just state hashes (ids) as unique identifiers to save memory
    let mut closed: HashSet<u64> = HashSet::new();

    // run LPT to get lower bound (LB) on the solution
    let (mut best, mut stats) = greedy_schedule(&processing_times, num_resources, start)?;
    // assume we can find optimal solution in time and adjust later if not
    stats.proved_optimal = true;
    stats.approx_factor = 1.;

    let mut queue = BinaryHeap::new();
    queue.push(BnBNode::init(num_resources, total_time, best.value));

    // Best-first search using f(N) = h(N)
    while let Some(node) = queue.pop() {
        stats.expanded += 1;

        // return current best if running for more than given time limit
        if let Some(timeout) = timeout {
            let elapsed = start.elapsed().expect("Failed to get elapsed system time.");
            if elapsed > timeout {
                stats.proved_optimal = false;
                stats.approx_factor = f64::nan();
                break;
            }
        }

        if node.schedule.len() == num_tasks {
            // complete solution
            if node.value.into_inner() < best.value {
                // found new minimum
                best = node.to_solution();
                stats.value = best.value;
            }
        } else {
            // inner node -> extend partial solution

            // select next task to schedule
            // heuristic: select the longest task that restricts the space the most
            //  - keep in mind that `processing_times` are already pre-sorted
            let task = *processing_times
                .iter()
                .find(|(task, _)| !node.schedule.contains_key(task))
                .expect("No task left yet solution was NOT complete!");

            // branch on pruned resources
            for (resource, _) in node.get_pruned_resources().into_iter() {
                // assign task -> resource
                match node.expand(task, resource, best.value, &closed) {
                    BnBExpansion::NewNode(n) => {
                        closed.insert(n.id);
                        queue.push(n)
                    }
                    BnBExpansion::ValueSubOptimal => stats.pruned_value += 1,
                    BnBExpansion::HeuristicSubOptimal => stats.pruned_h_value += 1,
                    BnBExpansion::StateVisited => stats.pruned_closed += 1,
                }
            }
        }
    }

    stats.elapsed = start.elapsed().expect("Failed to get elapsed system time.");
    Some((best, stats))
}

enum BnBExpansion<T: Float + Default> {
    NewNode(BnBNode<T>),
    ValueSubOptimal,
    HeuristicSubOptimal,
    StateVisited,
}

struct BnBNode<T>
where
    T: Float + Default,
{
    id: u64,
    schedule: HashMap<usize, usize>,
    completion_times: Vec<OrderedFloat<T>>,
    remaining_time: OrderedFloat<T>,
    value: OrderedFloat<T>,
    h_value: OrderedFloat<T>,
}

impl<T> BnBNode<T>
where
    T: Float + Default,
{
    fn compute_hash(completion_times: &[OrderedFloat<T>]) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut completion_times = completion_times.to_vec();
        completion_times.sort();
        completion_times.hash(&mut hasher);
        hasher.finish()
    }

    fn init(num_resources: usize, remaining_time: OrderedFloat<T>, h_value: T) -> Self {
        let completion_times = vec![OrderedFloat(T::zero()); num_resources];
        Self {
            id: Self::compute_hash(&completion_times),
            schedule: HashMap::new(), // TODO: with capacity `num_tasks`
            completion_times,
            remaining_time,
            value: OrderedFloat(T::zero()),
            h_value: OrderedFloat(h_value),
        }
    }

    fn get_pruned_resources(&self) -> Vec<(usize, OrderedFloat<T>)> {
        let mut resource_completions = self
            .completion_times
            .iter()
            .enumerate()
            .map(|(r, ct)| (r, *ct))
            .collect::<Vec<(usize, OrderedFloat<T>)>>();

        // prune resource symmetries:
        // if two resources have the same completion time, it's safely keep just one
        // because both trees are symmetric
        resource_completions.sort_by(|(_, x), (_, y)| x.cmp(y));
        resource_completions.dedup_by(|(_, x), (_, y)| x == y);

        resource_completions
    }

    fn expand(
        &self,
        task: (usize, OrderedFloat<T>),
        resource: usize,
        best_value: T,
        closed: &HashSet<u64>,
    ) -> BnBExpansion<T>
    where
        T: Float + Default,
    {
        let (task, pt) = task;
        let best_value = OrderedFloat(best_value);

        // generate core of the new state (completion times & remaining_time)
        let mut completion_times = self.completion_times.to_vec();
        completion_times[resource] = OrderedFloat(completion_times[resource].into_inner() + pt.0);

        let remaining_time = OrderedFloat(self.remaining_time.into_inner() - pt.into_inner());

        // evaluate objective fn for new schedule
        let value = *completion_times
            .iter()
            .max()
            .unwrap_or(&OrderedFloat(Float::infinity()));

        // prune sub-optimal
        if value < best_value || self.schedule.is_empty() {
            let id = BnBNode::compute_hash(&completion_times);

            // prune symmetries
            if closed.contains(&id) {
                return BnBExpansion::StateVisited;
            }

            let h_value = heuristic(&completion_times, remaining_time);
            return if h_value < best_value {
                let mut schedule = self.schedule.clone();
                schedule.insert(task, resource);
                BnBExpansion::NewNode(BnBNode {
                    id,
                    schedule,
                    completion_times,
                    remaining_time: OrderedFloat(remaining_time.into_inner()),
                    value,
                    h_value,
                })
            } else {
                BnBExpansion::HeuristicSubOptimal
            };
        }

        BnBExpansion::ValueSubOptimal
    }

    fn to_solution(&self) -> Solution<T> {
        // TODO: could this be done without copy?
        let mut schedule = vec![0usize; self.schedule.len()];
        for (task, resource) in self.schedule.iter() {
            schedule[*task] = *resource;
        }
        Solution {
            schedule,
            value: self.value.into_inner(),
            num_resources: self.completion_times.len(),
        }
    }
}

impl<T> Hash for BnBNode<T>
where
    T: Float + Default,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }

    fn hash_slice<H: Hasher>(data: &[Self], state: &mut H)
    where
        Self: Sized,
    {
        for s in data.iter() {
            s.id.hash(state);
        }
    }
}

impl<T> PartialEq for BnBNode<T>
where
    T: Float + Default,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for BnBNode<T> where T: Float + Default {}

impl<T> PartialOrd for BnBNode<T>
where
    T: Float + Default,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for BnBNode<T>
where
    T: Float + Default,
{
    fn cmp(&self, other: &BnBNode<T>) -> Ordering {
        // reverse because BnB uses a max-heap and our objective is to minimize
        self.h_value.cmp(&other.h_value).reverse()
    }
}

// TODO: Implement better heuristic
//  - allow task preemption but disallow parallel execution of task splits
//  - such heuristic should be still admissible and dominate this one (because it's more restricted)
fn heuristic<T>(
    completion_times: &[OrderedFloat<T>],
    remaining_time: OrderedFloat<T>,
) -> OrderedFloat<T>
where
    T: Float + Default,
{
    if completion_times.is_empty() {
        return OrderedFloat::default();
    }

    let mut completion_times = completion_times.to_vec();
    let mut remaining_time = remaining_time.into_inner();

    while OrderedFloat(remaining_time) > OrderedFloat(T::zero()) {
        let any_ct = completion_times.first().expect("empty completion times");

        if completion_times.iter().all(|ct| ct == any_ct) {
            let fraction = remaining_time / T::from::<usize>(completion_times.len()).unwrap();
            return OrderedFloat(any_ct.into_inner() + fraction);
        }

        let (a_min, min) = minimize(&completion_times, identity).unwrap();
        let (a_max, _) = minimize(&completion_times, |ct| OrderedFloat(ct.neg())).unwrap();

        let diff = completion_times[a_max].0 - min.0;
        completion_times[a_min] = OrderedFloat(min.0 + diff);
        remaining_time = remaining_time - diff;
    }

    completion_times.into_iter().max().unwrap()
}

#[inline]
fn compute_total_time<T>(processing_times: &[(usize, OrderedFloat<T>)]) -> OrderedFloat<T>
where
    T: Float + Sum,
{
    OrderedFloat(processing_times.iter().map(|(_, pt)| pt.into_inner()).sum())
}

#[cfg(test)]
mod test {
    use crate::alg::bnb;
    use ordered_float::OrderedFloat;

    #[test]
    fn heuristic() {
        let cts = vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(1.5)];

        // no remaining time
        assert_eq!(bnb::heuristic(&cts, OrderedFloat(0.)), OrderedFloat(2.));

        // some remaining time
        assert_eq!(bnb::heuristic(&cts, OrderedFloat(4.5)), OrderedFloat(3.));

        // another scenario with remaining time
        assert_eq!(bnb::heuristic(&cts, OrderedFloat(0.5)), OrderedFloat(2.));
    }
}
