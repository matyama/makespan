use criterion::{
    black_box, criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, SamplingMode,
};
use rand::prelude::*;

use makespan::mp;

const SEED: [u8; 16] = 123u128.to_le_bytes();

// Example processing times for 7 tasks
const EXAMPLE: [u32; 7] = [5, 5, 4, 4, 3, 3, 3];

fn identical_times<const T: u32>(n: usize) -> Vec<u32> {
    let mut pt = Vec::with_capacity(n);
    pt.fill(T);
    pt
}

fn subopt_instance<R: Rng + ?Sized>(num_resources: usize, rng: &mut R) -> Vec<u32> {
    let mut pts = vec![0; 2 * num_resources + 1];
    let mut pt = 2 * num_resources - 1;
    let mut i = 0;

    while pt >= num_resources {
        pts[i] = pt as u32;
        pts[i + 1] = pt as u32;
        pt -= 1;
        i += 2;
    }

    pts[i] = num_resources as u32;
    pts.shuffle(rng);

    pts
}

fn bench_mp_example(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("mp: varying resources with {EXAMPLE:?}"));

    for r in [3usize, 4].iter() {
        group.bench_with_input(BenchmarkId::new("lpt", r), r, |b, r| {
            b.iter(|| mp::lpt(black_box(&EXAMPLE), black_box(*r)))
        });

        group.bench_with_input(BenchmarkId::new("bnb", r), r, |b, r| {
            b.iter(|| mp::bnb(black_box(&EXAMPLE), black_box(*r)))
        });
    }

    group.finish();
}

fn bench_mp_identical_tasks(c: &mut Criterion) {
    let mut group = c.benchmark_group("mp: varying resources with identical tasks");

    for (r, n) in [(2usize, 7usize), (4, 7), (3, 50)].iter() {
        let p = identical_times::<1>(*n);

        group.bench_with_input(BenchmarkId::new("lpt", r), r, |b, r| {
            b.iter(|| mp::lpt(black_box(&p), black_box(*r)))
        });

        group.bench_with_input(BenchmarkId::new("bnb", r), r, |b, r| {
            b.iter(|| mp::bnb(black_box(&p), black_box(*r)))
        });
    }

    group.finish();
}

fn bench_mp_suboptimal_instance(c: &mut Criterion) {
    let mut rng = rand_pcg::Pcg64Mcg::from_seed(SEED);

    let mut group = c.benchmark_group("mp: varying resources with suboptimal instance");
    group.sampling_mode(SamplingMode::Flat);

    for r in (1..=3).map(|k| 10u32.pow(k) as usize) {
        group.bench_with_input(BenchmarkId::new("lpt", r), &r, |b, r| {
            b.iter_batched(
                || subopt_instance(*r, &mut rng),
                |p| mp::lpt(black_box(&p), black_box(*r)),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mp_example,
    bench_mp_identical_tasks,
    bench_mp_suboptimal_instance
);

criterion_main!(benches);
