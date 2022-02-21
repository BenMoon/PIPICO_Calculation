extern crate ndarray;
extern crate rand;
extern crate ndarray_rand;
extern crate sort;

use std::time::{Duration, Instant};
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::{arr1, array, s, Array, Array3};
use rand::distributions::{Distribution};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform as rand_uniform;
use ndhistogram::{Histogram, axis::Axis, ndhistogram, axis::Uniform, axis::Category, value::Mean};

fn genData(Nbins: usize, Nshots: usize, Npart: usize) -> Array2<f64> {
    // generate example data

    // initialize empty array
    let mut data = Array::<f64,_>::zeros((Nshots, Npart).f());
    // https://rust-lang-nursery.github.io/rust-cookbook/algorithms/randomness.html?highlight=random%20numb#generate-random-numbers
    let mut rng = rand::thread_rng();
    let dt1: rand::distributions::Uniform<f64> = rand::distributions::Uniform::from(-0.1..0.1);
    for i in 0..Nshots {
        let throw = dt1.sample(&mut rng);
        data[[i, 0]] = 3. - throw;
        data[[i, 1]] = 6. + throw;
        let a: ArrayBase<ndarray::OwnedRepr<f64>, Dim<[usize; 1]>> = Array::random(Npart-2, rand_uniform::new(0., 10.));
        //data[[i, 2:]] = 
        data.slice_mut(s![i, 2..]).assign(&a);

        // sort
        // check this https://github.com/rust-ndarray/ndarray/blob/master/examples/sort-axis.rs
        let mut vec: Vec<f64> = data.slice(s![i, ..]).to_vec();
        vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        data.slice_mut(s![i, ..]).assign(&Array::from_vec(vec));

        //println!("{} {} {}", i, throw, data.slice(s![i, ..]));
    }
    
    return data
}


fn main() {
    let nbins = 100;
    let Nshots = 10_000;
    let Npart = 10;

    let data = genData(nbins, Nshots, Npart);

    // sort data into histogram iterating through data 2D array
    // create a 2D histogram
    let mut hist = ndhistogram!(Uniform::<f64>::new(nbins, 0., 10.0), Uniform::<f64>::new(nbins, 0.,
    10.0));
    let (mut p1, mut p2) = (0, 0);
    let start = Instant::now();
    for i in 0..Nshots {
        p1 = 0;
        while p1 < Npart {
            p2 = p1 + 1;
            while p2 < Npart {
                hist.fill(&(data[[i, p1]], data[[i, p2]]));
                p2 += 1;
            }
            p1 += 1;
        }
    }
    let duration = start.elapsed();

    println!("{:?}", hist);
    println!("{:?}:", duration);


}