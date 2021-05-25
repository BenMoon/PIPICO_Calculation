extern crate itertools;
extern crate ndarray;
extern crate ndarray_stats;
extern crate noisy_float;

use ndarray_stats::{
    histogram::{strategies::Auto, Bins, Edges, Grid, GridBuilder, Histogram},
    HistogramExt,
};
use std::error::Error;
use std::result::Result;
use std::{ops::Index, vec};

//use ndarray::{Array1, Array3};
use itertools::Itertools;
use ndarray::{arr1, array, s, Array, Array3};
use noisy_float::types::{n64, N64};

fn calc_covariance() {
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

    /*
        bins = 5  # bins should as many as you have individual t's (in this case start to count a 0)
    correlated = np.zeros((bins, bins))
    for frame in data_to_corr:
        hist_1d_j = np.histogram(frame, bins=range(0, len(data_to_corr) + 1))[0]
        correlated += hist_1d_j[:, None] * hist_1d_j[None, :]
        */
}

fn hist_2d() -> Result<(), Box<dyn Error>> {
    let edges = Edges::from(vec![n64(-1.), n64(0.), n64(1.)]);
    let bins = Bins::new(edges);
    let square_grid = Grid::from(vec![bins.clone(), bins.clone()]);
    let mut histogram = Histogram::new(square_grid);

    let observation = array![n64(0.5), n64(0.6)];

    histogram.add_observation(&observation)?;

    let histogram_matrix = histogram.counts();
    let expected = array![[0, 0], [0, 1],];
    assert_eq!(histogram_matrix, expected.into_dyn());

    Ok(())
}

fn hist_1d() {
    // 1-dimensional observations, as a (n_observations, n_dimension) 2-d matrix
    let observations =
        Array::from_shape_vec((12, 1), vec![1, 4, 5, 2, 100, 20, 50, 65, 27, 40, 45, 23]).unwrap();

    // The optimal grid layout is inferred from the data, given a chosen strategy, Auto in this case
    let grid = GridBuilder::<Auto<usize>>::from_array(&observations)
        .unwrap()
        .build();

    let histogram = observations.histogram(grid);

    let histogram_matrix = histogram.counts();

    println!("{:?}", histogram_matrix);
}

fn hist_1d_fixed_bins() {
    // build histogram with fixed edges, 
    // value=edge doesn't fall into bin, compare: https://docs.rs/ndarray-stats/0.5.0/ndarray_stats/histogram/struct.Grid.html
    let observations =
        Array::from_shape_vec((12, 1), vec![1, 4, 5, 2, 100, 20, 50, 65, 27, 40, 45, 23]).unwrap();

    let edges = Edges::from(vec![1, 10, 20, 30, 50, 70, 90, 101]);
    let bins_x = Bins::new(edges);
    let grid = Grid::from(vec![bins_x]);

    let histogram = observations.histogram(grid);

    let histogram_matrix = histogram.counts();

    println!("{:?}", histogram_matrix);
}

fn main() {
    hist_1d_fixed_bins();
}
