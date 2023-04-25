
extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;
//extern crate sort;

use std::time::{Duration, Instant};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{arr1, array, s, Array, Array3};
use rand::Rng;
use rand::distributions::{Distribution};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform as rand_uniform;
use ndhistogram::{Histogram, axis::Axis, ndhistogram, axis::Uniform, axis::Category, value::Mean};

use polars::prelude::*;
use pyo3::prelude::*;

use pipico;
//use pipico::hallo;

fn gen_data(n_bins: usize, n_shots: usize, n_parts: usize) -> 
        (Array2<f64>,
         Array2<f64>,
         Array2<f64>) {
    // generate example data

    // initialize empty array
    let mut data_tof = Array::<f64,_>::zeros((n_shots, n_parts).f());
    let mut data_px  = Array::<f64,_>::zeros((n_shots, n_parts).f());
    let mut data_py  = Array::<f64,_>::zeros((n_shots, n_parts).f());
    // https://rust-lang-nursery.github.io/rust-cookbook/algorithms/randomness.html?highlight=random%20numb#generate-random-numbers
    let mut rng = rand::thread_rng();
    let dt1: rand::distributions::Uniform<f64> = rand::distributions::Uniform::from(-0.1..0.1);
    for i in 0..n_shots {
        let throw = dt1.sample(&mut rng);
        data_tof[[i, 0]] = 4. - throw;
        data_tof[[i, 1]] = 2. + throw;
        data_tof[[i, 2]] = 6. - throw;
        data_tof[[i, 3]] = 8. + throw;
        let a: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = Array::random(n_parts-4, rand_uniform::new(0., 10.));
        //data[[i, 4..]] = 
        data_tof.slice_mut(s![i as usize, 4usize..]).assign(&a);

        // sort
        // check this https://github.com/rust-ndarray/ndarray/blob/master/examples/sort-axis.rs
        //let mut vec: Vec<f64> = data_tof.slice(s![i, ..]).to_vec();
        //vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        //data_tof.slice_mut(s![i, ..]).assign(&Array::from_vec(vec));
    }

    for i in 0..n_shots {
        let throw = dt1.sample(&mut rng);
        data_px[[i, 0]] = throw;
        data_py[[i, 0]] = throw;
        data_px[[i, 1]] = -throw;
        data_py[[i, 1]] = -throw;

        let throw = dt1.sample(&mut rng);
        data_px[[i, 2]] = throw;
        data_py[[i, 2]] = throw;
        data_px[[i, 3]] = -throw;
        data_py[[i, 3]] = -throw;

        let a: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = Array::random(n_parts-4, rand_uniform::new(0., 10.));
        data_px.slice_mut(s![i as usize, 4usize..]).assign(&a);
        let a: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = Array::random(n_parts-4, rand_uniform::new(0., 10.));
        data_py.slice_mut(s![i as usize, 4usize..]).assign(&a);
    }

    
    return (data_tof, data_px, data_py)
}


fn main() {
    let n_bins = 10;
    let n_shots = 4;
    let n_parts = 10;

    let (data_tof, data_px, data_py) = gen_data(n_bins, n_shots, n_parts);


    //let trigger_nrs = (0..n_shots).into_iter().map(|x|x);
    let mut trigger_nrs = Vec::new();
    for i in (0..n_shots).into_iter() {
        for j in (0..n_parts).into_iter() {
            trigger_nrs.push(i);
        }
    }

    // open file
    let mut file = std::fs::File::open("/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data-1e3-1e1.parquet").expect("file not found");
    // read to DataFrame
    let df = ParquetReader::new(&mut file).finish().unwrap();
    //dbg!(&df);

    let a = df.select(["trigger nr", "tof", "px", "py"]).unwrap().to_ndarray::<Float64Type>().unwrap();
    //pipico::polars_filter_momentum_bench_idx(a.clone());
    //dbg!(a);

    let b = pipico::ndarray_filter_momentum_bench_2D(a.clone());
    //dbg!(b);
    //println!("{:?}", b);
 
    let a = df.select(["trigger nr", "idx", "px", "py", "pz"]).unwrap().to_ndarray::<Float64Type>().unwrap();
    let (fg, bg) = pipico::get_pairs_bench(a.clone());
    dbg!(fg, bg);
    //println!("{:?}", c);

    //let c = pipico::polars_filter_momentum_bench_idx(a.clone());
    //dbg!(c);

    //let mut rng = rand::thread_rng();
    //pipico::get_bg_idx(&mut rng);
    //pipico::get_bg_idx(rng);


    //pyo3::prepare_freethreaded_python();
    //Python::with_gil(|py| py.run("print('Hello World')", None, None));
    //Python::with_gil(|py| py.run(
     //   "import sys; print('hallo')", None, None));
    

    /*
    // sort data into histogram iterating through data 2D array
    // create a 2D histogram
    let mut hist = ndhistogram!(Uniform::<f64>::new(n_bins, 0., 10.0), Uniform::<f64>::new(n_bins, 0.,
    10.0));
    let (mut p1, mut p2) = (0, 0);
    let start = Instant::now();
    for i in 0..n_shots {
        p1 = 0;
        while p1 < n_parts {
            p2 = p1 + 1;
            while p2 < n_parts {
                hist.fill(&(data[[i, p1]], data[[i, p2]]));
                p2 += 1;
            }
            p1 += 1;
        }
    } 
    */
    //let duration = start.elapsed();

    //println!("{:?}", hist);
    //println!("{:?}:", duration);


}