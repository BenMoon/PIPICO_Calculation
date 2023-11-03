extern crate ndarray;
extern crate ndarray_rand;
extern crate rand;

use ndarray::prelude::*;

use polars::prelude::*;

fn main() {
    //let n_bins = 10;
    let n_shots = 4;
    let n_parts = 10;

    //let trigger_nrs = (0..n_shots).into_iter().map(|x|x);
    let mut trigger_nrs = Vec::new();
    for i in 0..n_shots {
        for _ in 0..n_parts {
            trigger_nrs.push(i);
        }
    }

    // open file
    let file = std::fs::File::open(
        "/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data-1e3-1e1.feather",
    )
    .expect("file not found");
    // read to DataFrame
    let df = IpcReader::new(file).finish().unwrap();
    dbg!(&df);

    let a = df
        .select(["trigger nr", "idx", "px", "py", "pz"])
        .unwrap()
        .to_ndarray::<Float64Type>(IndexOrder::C)
        .unwrap();
    let (fg, bg) = pipico::get_pairs_bench(a.clone());
    dbg!(fg, bg);

    let a = arr2(&[
        [1., 1., 1., 1., 0., 4., 18.],
        [1., 2., -1., -1., 0., 4., 18.],
        [2., 3., 3., 1., 0., 4., 18.],
        [2., 4., -1., -1., 0., 4., 18.],
        [2., 5., 1., 1., 0., 4.1, 15.],
        [2., 6., -1., -1., 0., 4.1, 15.],
        [3., 7., 2., 2., 0., 4.1, 15.],
        [3., 8., -2., -2., 0., 4.1, 15.],
        [4., 9., 3., 3., 0., 4.1, 15.],
        [4., 10., 0., 0., 0., 4.1, 15.],
        [6., 11., -1., -1., 0., 4.1, 15.],
    ]);
    let mass_momentum_cut = arr2(&[
        [17.5, 18.5, 17.5, 18.5, 1.],
        [16.5, 17.5, 17.5, 18.5, 1.],
        [16.5, 17.5, 18.5, 19.5, 1.],
    ]);
    let default_momentum_cut = 1.;
    dbg!(&default_momentum_cut);
    let (fg, _bg) = pipico::get_covar_pairs_fixed_cut(a, mass_momentum_cut, default_momentum_cut);
    dbg!(fg);
}
