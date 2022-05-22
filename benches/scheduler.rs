// FIXME: use criterion
#![feature(test)]

use rand::prelude::*;

extern crate test;
use test::Bencher;

extern crate makespan;
use makespan::mp;

const PROCESSING_TIMES: [u32; 7] = [5, 5, 4, 4, 3, 3, 3];
const IDENTICAL_TIMES: [u32; 7] = [1; 7];
const IDENTICAL_TIMES_LONG: [u32; 50] = [1; 50];

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

// TODO: Deduplicate with macro

#[bench]
fn lpt_3_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| mp::lpt(&PROCESSING_TIMES, 3));
}

#[bench]
fn bnb_3_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| mp::bnb(&PROCESSING_TIMES, 3));
}

#[bench]
fn lpt_4_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| mp::lpt(&PROCESSING_TIMES, 4));
}

#[bench]
fn bnb_4_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| mp::bnb(&PROCESSING_TIMES, 4));
}

#[bench]
fn lpt_2_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| mp::lpt(&IDENTICAL_TIMES, 4));
}

#[bench]
fn bnb_2_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| mp::bnb(&IDENTICAL_TIMES, 4));
}

#[bench]
fn lpt_4_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| mp::lpt(&IDENTICAL_TIMES, 4));
}

#[bench]
fn bnb_4_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| mp::bnb(&IDENTICAL_TIMES, 4));
}

#[bench]
fn lpt_3_resources_50_identical_tasks(b: &mut Bencher) {
    b.iter(|| mp::lpt(&IDENTICAL_TIMES_LONG, 3));
}

#[bench]
fn bnb_3_resources_50_identical_tasks(b: &mut Bencher) {
    b.iter(|| mp::bnb(&IDENTICAL_TIMES_LONG, 3));
}

#[bench]
fn bnb_3_resources_50_identical_tasks_longer(b: &mut Bencher) {
    let pts = vec![10u32; 50];
    b.iter(|| mp::bnb(&pts, 3));
}

#[bench]
fn lpt_subopt_10(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let num_resources = 10;
    let processing_times = subopt_instance(num_resources, &mut rng);
    b.iter(|| mp::lpt(&processing_times, num_resources));
}

#[bench]
fn lpt_subopt_100(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let num_resources = 100;
    let processing_times = subopt_instance(num_resources, &mut rng);
    b.iter(|| mp::lpt(&processing_times, num_resources));
}

#[bench]
fn lpt_subopt_1000(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let num_resources = 1000;
    let processing_times = subopt_instance(num_resources, &mut rng);
    b.iter(|| mp::lpt(&processing_times, num_resources));
}
