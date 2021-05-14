extern crate itertools;
extern crate ndarray;

use std::{ops::Index, vec};

//use ndarray::{Array1, Array3};
use itertools::Itertools;
use ndarray::arr1;
use ndarray::s;
use ndarray::Array3;

fn main() {
    // initialize dummy values
    let x = arr1(&[25, 25, 30, 35, 25, 40, 35, 30]);
    let y = arr1(&[110, 115, 130, 125, 115, 120, 115, 120]);
    let t = arr1(&[1, 2, 3, 4, 2, 3, 4, 2]);
    let ids = arr1(&[0, 0, 1, 1, 1, 2, 3, 4]);

    // helpful discussion on unique
    // https://datacrayon.com/posts/programming/rust-notebooks/unique-array-elements-and-their-frequency/
    let nr_triggers = ids.iter().unique().count(); // how many unique trigger we have
    let unique_triggers = ids.iter().cloned().unique().collect_vec();
    let mut unique_frequency: Vec<usize> = Vec::<usize>::new();
    for unique_elem in unique_triggers.iter() {
        unique_frequency.push(ids.iter().filter(|&elem| elem == unique_elem).count());
    }

    // build n-dimensional array for future advanced features
    let max_trig_len = *unique_frequency.iter().max().unwrap();
    let mut data = Array3::<i32>::ones([nr_triggers, max_trig_len, 3]);
    data *= -1;
    for trigger_num in unique_triggers {
        for (i, val) in ids.iter().positions(|&x| x == trigger_num).enumerate() {
            data[[trigger_num, i, 0]] = x[val];
            data[[trigger_num, i, 1]] = y[val];
            data[[trigger_num, i, 2]] = t[val];
        }
    }
    let data_to_correlate = data.slice(s![.., .., 2]);
}
