// cargo flamegraph --root --bench my_benchmark -- --bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pipico;

use polars::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    // open file
    let mut file = std::fs::File::open(
        "/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data.parquet",
    )
    .unwrap();
    // read to DataFrame
    let df = ParquetReader::new(&mut file).finish().unwrap();
    dbg!(df.shape());

    let a = df
        .select(["trigger nr", "tof", "px", "py"])
        .unwrap()
        .to_ndarray::<Float64Type>()
        .unwrap();
    c.bench_function("numpy 2D array", |b| {
        b.iter(|| pipico::polars_filter_momentum_bench_2D(black_box(a.clone())))
    });
    c.bench_function("numpy index", |b| {
        b.iter(|| pipico::polars_filter_momentum_bench_idx(black_box(a.clone())))
    });
}

// benchmark get background with index
pub fn criterion_get_bg_idx(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    c.bench_function("for loop comparison", |b| {
        b.iter(|| pipico::get_bg_idx(black_box(&mut rng)))
    });
}

criterion_group!(benches, criterion_get_bg_idx);
//criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
