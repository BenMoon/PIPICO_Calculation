// cargo flamegraph --root --bench my_benchmark -- --bench

use std::thread;
use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use itertools::Itertools;
use ndarray::{s, Array, Axis};

use polars::prelude::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    // open file
    let file = std::fs::File::open(
        "/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data-1e3-1e1.feather",
    )
    .expect("File not found");
    // read to DataFrame
    let df = IpcReader::new(file).finish().unwrap();
    dbg!(df.shape());

    // let df_tof = df
    //     .select(["trigger nr", "tof", "px", "py", "pz"])
    //     .unwrap()
    //     .to_ndarray::<Float64Type>(IndexOrder::C)
    //     .unwrap();

    let df_idx = df
        .select(["trigger nr", "idx", "px", "py", "pz"])
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();

    c.bench_function("par_iter idx", |b| {
        b.iter(|| pipico::get_pairs_bench(black_box(df_idx.clone())))
    });
    thread::sleep(Duration::from_secs(10));
}

// benchmark get background with index
pub fn criterion_get_bg_idx(c: &mut Criterion) {
    let mut rng = rand::thread_rng();

    // all this boilerplate is a little stupid, but right now I don't know how I would otherwhise get
    // ArrayBase<ViewRepr<&&&f64>, Dim<[usize; 1]>>
    // required for the get_bg_idx
    let file = std::fs::File::open(
        "/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data-1e3-1e1.feather",
    )
    .expect("file not found");
    // read to DataFrame
    let df = IpcReader::new(file).finish().unwrap();

    //let trigger_frame = Array::from_shape_vec((100, 2), (100..300).into_iter().map(|x| x as f64).collect_vec()).expect("shape incorrect");
    let data = df
        .select(["trigger nr", "idx", "px", "py"])
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();

    let chunk_vec = data.axis_iter(Axis(0)).flatten().collect_vec();
    let data_chunk =
        Array::from_shape_vec((data.len() / data.ncols(), data.ncols()), chunk_vec).unwrap();
    let trigger_frame_vec = data_chunk
        .axis_iter(Axis(0))
        .filter(|x| *x[0] == 100.)
        .flatten()
        .collect_vec();
    let trigger_frame = Array::from_shape_vec(
        (trigger_frame_vec.len() / data.ncols(), data.ncols()),
        trigger_frame_vec,
    )
    .unwrap();
    let trg_frame_indizes = trigger_frame.slice(s![.., 1]);
    dbg!(&trg_frame_indizes.iter().minmax());

    c.bench_function("gen bg: for loop comparison", |b| {
        b.iter(|| {
            pipico::get_bg_idx(
                black_box(&mut rng),
                black_box(trg_frame_indizes),
                black_box(data.nrows()),
            )
        })
    });
    c.bench_function("gen bg: HashSet", |b| {
        b.iter(|| {
            pipico::get_bg_idx_set(
                black_box(&mut rng),
                black_box(trg_frame_indizes),
                black_box(data.nrows()),
            )
        })
    });
    c.bench_function("gen bg: BitSet", |b| {
        b.iter(|| {
            pipico::get_bg_idx_set_optimized(
                black_box(&mut rng),
                black_box(trg_frame_indizes),
                black_box(data.nrows()),
            )
        })
    });
}

// run the benchmarks
criterion_group!(benches, criterion_benchmark, criterion_get_bg_idx);
criterion_main!(benches);
