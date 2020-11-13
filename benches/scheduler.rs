#![feature(test)]

use rand::prelude::*;

extern crate test;
use test::Bencher;

extern crate makespan;
use makespan::Scheduler;

const LPT: Scheduler = Scheduler::LPT;
const BNB: Scheduler = Scheduler::BnB { timeout: None };

const PROCESSING_TIMES: [f64; 7] = [5., 5., 4., 4., 3., 3., 3.];
const IDENTICAL_TIMES: [f64; 7] = [1.; 7];
const IDENTICAL_TIMES_LONG: [f64; 50] = [1.; 50];

// TODO: should rather return `Vec<usize>`
fn subopt_instance<R: Rng + ?Sized>(num_resources: usize, rng: &mut R) -> Vec<f64> {
    let mut pts = vec![0usize; 2 * num_resources + 1];
    let mut pt = 2 * num_resources - 1;
    let mut i = 0;
    while pt >= num_resources {
        pts[i] = pt;
        pts[i + 1] = pt;
        pt -= 1;
        i += 2;
    }
    pts[i] = num_resources;
    pts.shuffle(rng);
    pts.into_iter().map(|pt| pt as f64).collect()
}

// TODO: Deduplicate with macro

#[bench]
fn lpt_3_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| LPT.schedule(&PROCESSING_TIMES, 3));
}

#[bench]
fn bnb_3_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| BNB.schedule(&PROCESSING_TIMES, 3));
}

#[bench]
fn lpt_4_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| LPT.schedule(&PROCESSING_TIMES, 4));
}

#[bench]
fn bnb_4_resources_7_tasks(b: &mut Bencher) {
    b.iter(|| BNB.schedule(&PROCESSING_TIMES, 4));
}

#[bench]
fn lpt_2_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| LPT.schedule(&IDENTICAL_TIMES, 4));
}

#[bench]
fn bnb_2_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| BNB.schedule(&IDENTICAL_TIMES, 4));
}

#[bench]
fn lpt_4_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| LPT.schedule(&IDENTICAL_TIMES, 4));
}

#[bench]
fn bnb_4_resources_7_identical_tasks(b: &mut Bencher) {
    b.iter(|| BNB.schedule(&IDENTICAL_TIMES, 4));
}

#[bench]
fn lpt_3_resources_50_identical_tasks(b: &mut Bencher) {
    b.iter(|| LPT.schedule(&IDENTICAL_TIMES_LONG, 3));
}

#[bench]
fn bnb_3_resources_50_identical_tasks(b: &mut Bencher) {
    b.iter(|| BNB.schedule(&IDENTICAL_TIMES_LONG, 3));
}

#[bench]
fn bnb_3_resources_50_identical_tasks_longer(b: &mut Bencher) {
    let pts = vec![10.; 50];
    b.iter(|| BNB.schedule(&pts, 3));
}

#[bench]
fn lpt_subopt_10(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let num_resources = 10;
    let processing_times = subopt_instance(num_resources, &mut rng);
    b.iter(|| LPT.schedule(&processing_times, num_resources));
}

#[bench]
fn lpt_subopt_100(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let num_resources = 100;
    let processing_times = subopt_instance(num_resources, &mut rng);
    b.iter(|| LPT.schedule(&processing_times, num_resources));
}

#[bench]
fn lpt_subopt_1000(b: &mut Bencher) {
    let mut rng = rand::thread_rng();
    let num_resources = 1000;
    let processing_times = subopt_instance(num_resources, &mut rng);
    b.iter(|| LPT.schedule(&processing_times, num_resources));
}
