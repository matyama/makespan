//! # Makespan
//! This Rust library implements several solvers for task scheduling problems optimizing
//! [makespan](https://en.wikipedia.org/wiki/Makespan).
//!
//! ## Graham's notation
//! Definitions of scheduling problems in *operations research* commonly use
//! [Graham's notation](https://en.wikipedia.org/wiki/Optimal_job_scheduling) `a|b|c` where
//!   1. `a` defines *machine environment* (resources)
//!   2. `b` denotes *task/job characteristics*
//!   3. `c` specifies the *objective function* that is to be minimized (note that this library
//!      focuses on optimizing the *makespan* `C_max`)
//!
//! # Covered scheduling problems
//! The module structure of this library follows a structure of the general taxonomy of scheduling
//! problems and as mentioned above focuses on minimizing the *makespan* (`C_max`).
//!
//! General constraints:
//!  - Each task can be processed by at most one resource at a time
//!  - Each resource can process at most one task at a time
//!
//! ## Single-processor
//! This class of scheduling problems considers single processing unit (resource).
//! NOTE: not implemented yet
//!
//! ## Multi-processor
//! This class of scheduling problems considers multiple processing units (resources).
//!
//! ### Non-preemptive
//! This sub-class considers non-preemptive tasks and is covered by module [`mp`](src/mp.rs). The
//! list of problems and corresponding algorithms includes:
//!  - `P||C_max`: LPT (approx), BnB (optimal)
//!
//! ### Preemptive
//! This sub-class considers preemptive tasks and is covered by module [`mp_pmtn`](src/mp_pmtn.rs).
//! The list of problems and corresponding algorithms includes:
//!  - `P|pmtn|C_max`: McNaughton (optimal)
//!
//! # Resources
//!  - [Complexity results](http://www2.informatik.uni-osnabrueck.de/knust/class/dateien/allResults.pdf)
//!  - [The scheduling zoo](http://www-desir.lip6.fr/~durrc/query/)
//!  - [CTU lecture slides](https://rtime.ciirc.cvut.cz/~hanzalek/KO/sched_e.pdf)
use num_traits::{Num, NumCast};
use ordered_float::{Float, NotNan, OrderedFloat};

pub mod mp;
pub mod mp_pmtn;

/// Class of types representing a totally ordered numerical time.
///
/// The most significant features of [Time] include:
///  1. the type must support basic numerical arithmetic via [Num] and conversions through [NumCast]
///  1. this type must also be [Eq] (comparable on equality) and [Ord] (totally ordered)
///  1. there is a maximal value representing the notion of infinity
///
/// This library includes implmentations for standard unsigned integral types. For floats one can
/// use for instance [OrderedFloat] or [NotNan] types which extend [`f64`]/[`f32`] to [`Ord`].
pub trait Time: Num + Copy + NumCast + Ord + Eq {
    /// Value representing time in the infinite future.
    fn inf() -> Self;

    /// Tests if this value represents the maximal value (infinity).
    fn is_inf(&self) -> bool {
        self == &Self::inf()
    }
}

// TODO: this is only used in tests so unless we want to provide this blanket impl, the
// `ordered_float` dependency could be moved to dev and this under a `#[cfg(test)]`
/// Blanket implementation of [Time] for [OrderedFloat] for any [Float] type.
impl<T: Float> Time for OrderedFloat<T> {
    #[inline]
    fn inf() -> Self {
        T::infinity().into()
    }
}

/// Blanket implementation of [Time] for [NotNan] for any [Float] type.
impl<T: Float> Time for NotNan<T> {
    #[inline]
    fn inf() -> Self {
        NotNan::new(T::infinity()).expect("infinity is not nan")
    }
}

macro_rules! time_impl {
    ($t:ty) => {
        impl Time for $t {
            #[inline]
            fn inf() -> Self {
                Self::MAX
            }
        }
    };
}

time_impl!(u8);
time_impl!(u16);
time_impl!(u32);
time_impl!(u64);
time_impl!(u128);
