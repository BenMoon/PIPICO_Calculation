// cargo flamegraph --root --bench my_benchmark -- --bench

use std::time::Duration;
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pipico;

use polars::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    // open file
    let mut file = std::fs::File::open(
        "/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data-1e4-1e2.parquet",
    )
    .expect("File not found");
    // read to DataFrame
    let df = ParquetReader::new(&mut file).finish().unwrap();
    dbg!(df.shape());

    let df_tof = df
        .select(["trigger nr", "tof", "px", "py"])
        .unwrap()
        .to_ndarray::<Float64Type>()
        .unwrap();

    let df_idx = df
        .select(["trigger nr", "idx", "px", "py"])
        .unwrap()
        .to_ndarray::<Float64Type>()
        .unwrap();
    
    c.bench_function("par_iter inner", |b| {
        b.iter(|| pipico::ndarray_filter_momentum_bench_2D(black_box(df_tof.clone())))
    });
    thread::sleep(Duration::from_secs(10));

    c.bench_function("par_iter idx", |b| {
        b.iter(|| pipico::ndarray_filter_momentum_bench_idx(black_box(df_idx.clone())))
    });
    thread::sleep(Duration::from_secs(10));
    
    c.bench_function("par_iter outer", |b| {
       b.iter(|| pipico::ndarray_filter_momentum_bench_par_outer(black_box(df_tof.clone())))
    });
    thread::sleep(Duration::from_secs(10));
    
    c.bench_function("2D array: outer for", |b| {
       b.iter(|| pipico::ndarray_filter_momentum_bench_outer_for(black_box(df_tof.clone())))
    });

}

// benchmark get background with index
pub fn criterion_get_bg_idx(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    c.bench_function("gen bg: for loop comparison", |b| {
        b.iter(|| pipico::get_bg_idx(black_box(&mut rng)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
