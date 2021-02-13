use std::cmp::{max, Ordering};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::convert::identity;
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::time::{Duration, SystemTime};

use num_traits::Float;
use ordered_float::OrderedFloat;
use voracious_radix_sort::RadixSort;

use crate::alg::core::*;
use crate::alg::lpt::greedy_schedule;

/// Search for optimal solution of `P || C_max` using Best-first BnB.
///
/// The *heuristic* `h(N)` of a node `N` (partial solution) is the minimum makespan of a schedule
/// that is initialized to the partial solution of `N` and completed by relaxing the problem and
/// allowing task preemption (i.e. remaining tasks are scheduled as `P |pmtn| C_max`).
pub(crate) fn bnb<T, H>(
    processing_times: &[T],
    num_resources: usize,
    timeout: Option<Duration>,
    heuristic: &H,
) -> Option<(Solution<T>, Stats<T>)>
where
    T: Float + Sum,
    H: Heuristic<T>,
{
    let start = SystemTime::now();

    // TODO: deduplicate
    if processing_times.is_empty() || num_resources == 0 {
        return None;
    }

    let num_tasks = processing_times.len();

    let remaining_times = processing_times;
    let mut tasks = preprocess(processing_times);

    // pre-sorted set of tasks:
    //  1. LPT does it anyway
    //  2. Heuristic: larger tasks might prude the space sooner
    tasks.voracious_sort();
    tasks.reverse();

    // closed set will store just state hashes (ids) as unique identifiers to save memory
    let mut closed: HashSet<u64> = HashSet::new();

    // run LPT to get lower bound (LB) on the solution
    let (mut best, mut stats) = greedy_schedule(&tasks, num_resources, start)?;
    // assume we can find optimal solution in time and adjust later if not
    stats.proved_optimal = true;
    stats.approx_factor = 1.;

    let mut queue = BinaryHeap::new();
    queue.push(BnBNode::init(num_resources, &remaining_times, best.value));

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
                best = node.into_solution();
                stats.value = best.value;
            }
        } else {
            // inner node -> extend partial solution

            // select next task to schedule
            // heuristic: select the longest task that restricts the space the most
            //  - keep in mind that `processing_times` are already pre-sorted
            let task = tasks
                .iter()
                .find(|Task { id, .. }| !node.schedule.contains_key(id))
                .expect("No task left yet solution was NOT complete!"); // TODO: return Result

            // branch on pruned resources
            for (resource, _) in node.get_pruned_resources().into_iter() {
                // assign task -> resource
                match node.expand(task, resource, best.value, &closed, heuristic) {
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

enum BnBExpansion<T: Float> {
    NewNode(BnBNode<T>),
    ValueSubOptimal,
    HeuristicSubOptimal,
    StateVisited,
}

struct BnBNode<T: Float> {
    id: u64,
    schedule: HashMap<usize, usize>,
    completion_times: Vec<OrderedFloat<T>>,
    remaining_times: Vec<OrderedFloat<T>>,
    value: OrderedFloat<T>,
    h_value: OrderedFloat<T>,
}

impl<T: Float> BnBNode<T> {
    fn compute_hash(completion_times: &[OrderedFloat<T>]) -> u64 {
        let mut hasher = DefaultHasher::new();
        let mut completion_times = completion_times.to_vec();
        completion_times.sort();
        completion_times.hash(&mut hasher);
        hasher.finish()
    }

    fn init(num_resources: usize, remaining_times: &[T], h_value: T) -> Self {
        let completion_times = vec![OrderedFloat(T::zero()); num_resources];
        Self {
            id: Self::compute_hash(&completion_times),
            schedule: HashMap::new(), // TODO: with capacity `num_tasks`
            completion_times,
            remaining_times: remaining_times.iter().map(|t| OrderedFloat(*t)).collect(),
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
        // if two resources have the same completion time, it's safe to keep just one
        // because both trees are symmetric
        resource_completions.sort_by(|(_, x), (_, y)| x.cmp(y));
        resource_completions.dedup_by(|(_, x), (_, y)| x == y);

        resource_completions
    }

    fn expand<H>(
        &self,
        task: &Task<T>,
        resource: usize,
        best_value: T,
        closed: &HashSet<u64>,
        heuristic: &H,
    ) -> BnBExpansion<T>
    where
        T: Float,
        H: Heuristic<T>,
    {
        let best_value = OrderedFloat(best_value);

        // generate core of the new state (completion times & remaining_time)
        let mut completion_times = self.completion_times.to_vec();
        completion_times[resource] = completion_times[resource] + task.pt;

        // evaluate objective fn for new schedule
        let value = *completion_times
            .iter()
            .max()
            .unwrap_or(&T::infinity().into());

        // prune sub-optimal
        if value < best_value || self.schedule.is_empty() {
            let id = BnBNode::compute_hash(&completion_times);

            // prune symmetries
            if closed.contains(&id) {
                return BnBExpansion::StateVisited;
            }

            let mut remaining_times = self.remaining_times.clone();
            remaining_times[resource] = remaining_times[resource] - task.pt;

            let h_value = heuristic.eval(&completion_times, &remaining_times);

            return if h_value < best_value {
                let mut schedule = self.schedule.clone();
                schedule.insert(task.id, resource);
                BnBExpansion::NewNode(BnBNode {
                    id,
                    schedule,
                    completion_times,
                    remaining_times,
                    value,
                    h_value,
                })
            } else {
                BnBExpansion::HeuristicSubOptimal
            };
        }

        BnBExpansion::ValueSubOptimal
    }

    fn into_solution(self) -> Solution<T> {
        let mut schedule = vec![0usize; self.schedule.len()];
        for (task, resource) in self.schedule.into_iter() {
            schedule[task] = resource;
        }
        Solution {
            schedule,
            value: self.value.into_inner(),
            num_resources: self.completion_times.len(),
        }
    }
}

impl<T: Float> Hash for BnBNode<T> {
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

impl<T: Float> PartialEq for BnBNode<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T: Float> Eq for BnBNode<T> {}

impl<T: Float> PartialOrd for BnBNode<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Float> Ord for BnBNode<T> {
    fn cmp(&self, other: &BnBNode<T>) -> Ordering {
        // reverse because BnB uses a max-heap and our objective is to minimize
        self.h_value.cmp(&other.h_value).reverse()
    }
}

pub(crate) trait Heuristic<T: Float> {
    fn eval(
        &self,
        completion_times: &[OrderedFloat<T>],
        remaining_times: &[OrderedFloat<T>],
    ) -> OrderedFloat<T>;
}

pub(crate) struct PreemptionHeuristic;

impl<T: Float + Sum> Heuristic<T> for PreemptionHeuristic {
    fn eval(
        &self,
        completion_times: &[OrderedFloat<T>],
        remaining_times: &[OrderedFloat<T>],
    ) -> OrderedFloat<T> {
        let num_resources: OrderedFloat<T> = T::from(completion_times.len())
            .unwrap_or_else(T::one)
            .into();

        let mut sum_remaining: OrderedFloat<T> = T::zero().into();
        let mut max_remaining: OrderedFloat<T> = T::zero().into();

        for &t in remaining_times.iter() {
            sum_remaining = sum_remaining + t;
            max_remaining = max(max_remaining, t);
        }

        let ideally_parallelized = (sum(completion_times) + sum_remaining) / num_resources;

        max(max_remaining, ideally_parallelized)
    }
}

pub(crate) struct FullPreemptionHeuristic;

impl<T: Float + Sum> Heuristic<T> for FullPreemptionHeuristic {
    fn eval(
        &self,
        completion_times: &[OrderedFloat<T>],
        remaining_times: &[OrderedFloat<T>],
    ) -> OrderedFloat<T> {
        if completion_times.is_empty() {
            return T::zero().into();
        }

        let mut completion_times = completion_times.to_vec();
        let mut remaining_time = sum(remaining_times);

        let zero = T::zero().into();
        while remaining_time > zero {
            let any_ct = completion_times.first().expect("empty completion times");

            if completion_times.iter().all(|ct| ct == any_ct) {
                let num_resources: OrderedFloat<T> = T::from(completion_times.len())
                    .unwrap_or_else(T::one)
                    .into();
                return *any_ct + remaining_time / num_resources;
            }

            let (a_min, min) = minimize(&completion_times, identity).unwrap();
            let (a_max, _) = minimize(&completion_times, |ct| ct.neg().into()).unwrap();

            let diff = completion_times[a_max] - min;
            completion_times[a_min] = min + diff;
            remaining_time = remaining_time - diff;
        }

        completion_times
            .into_iter()
            .max()
            .unwrap_or_else(|| T::zero().into())
    }
}

// TODO: Add heuristic to this config
// TODO: Generalize bound on `T`
pub(crate) struct BranchAndBound {
    pub(crate) timeout: Option<Duration>,
}

// TODO: Replace `Float` by more appropriate trait(s)
impl<T: Float + Sum> Solve<T> for BranchAndBound {
    fn solve(&self, processing_times: &[Self::Time], num_resources: usize) -> Option<Self::Output> {
        bnb(
            processing_times,
            num_resources,
            self.timeout,
            &PreemptionHeuristic,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::alg::bnb;
    use crate::alg::bnb::Heuristic;
    use ordered_float::OrderedFloat;

    #[test]
    fn fully_preemptive_heuristic() {
        let h = bnb::FullPreemptionHeuristic;

        let cts = vec![OrderedFloat(1.0), OrderedFloat(2.0), OrderedFloat(1.5)];

        // no remaining time
        let remaining_times = vec![OrderedFloat(0.), OrderedFloat(0.), OrderedFloat(0.)];
        assert_eq!(h.eval(&cts, &remaining_times), OrderedFloat(2.));

        // some remaining time
        let remaining_times = vec![OrderedFloat(1.5), OrderedFloat(2.0), OrderedFloat(1.0)];
        assert_eq!(h.eval(&cts, &remaining_times), OrderedFloat(3.));

        // another scenario with remaining time
        let remaining_times = vec![OrderedFloat(0.5), OrderedFloat(0.0), OrderedFloat(0.0)];
        assert_eq!(h.eval(&cts, &remaining_times), OrderedFloat(2.));
    }
}
