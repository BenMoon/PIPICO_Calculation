
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pipico::polars_filter_momentum_bench;

use polars::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    // open file
    let mut file = std::fs::File::open("/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data.parquet").unwrap();
    // read to DataFrame
    let df = ParquetReader::new(&mut file).finish().unwrap();

    let a = df.select(["trigger nr", "tof", "px", "py"]).unwrap().to_ndarray().unwrap();
    c.bench_function("numpy 1000", |b| b.iter(|| polars_filter_momentum_bench(black_box(df.clone()))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
