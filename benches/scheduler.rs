#![feature(test)]

extern crate test;
use test::Bencher;

extern crate makespan;
use makespan::Scheduler;

const LPT: Scheduler = Scheduler::LPT;
const BNB: Scheduler = Scheduler::BnB { timeout: None };

const PROCESSING_TIMES: [f64; 7] = [5., 5., 4., 4., 3., 3., 3.];
const IDENTICAL_TIMES: [f64; 7] = [1.; 7];
const IDENTICAL_TIMES_LONG: [f64; 50] = [1.; 50];

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
