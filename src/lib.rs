#![feature(iter_collect_into)]

use pyo3::ffi::PyImport_ImportModuleEx;
use rand::rngs::ThreadRng;
use rand::{self, Rng};
use std::ops::Add;
use std::usize;

use itertools::{Itertools, izip};
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{s, Array1, Array2, Data};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use ndhistogram::{axis::Uniform, ndhistogram, Histogram};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
//use polars::export::arrow::compute::filter;
//use polars::lazy::dsl::apply_multiple;
use pyo3::{pymodule, types::PyModule, PyResult, Python};
extern crate num_cpus;

//use polars::prelude::*;
//use pyo3_polars::error::PyPolarsErr;
//use pyo3_polars::PyDataFrame;

use rayon::prelude::*;

/// calculate a covariance map
#[pymodule]
fn pipico(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// calculate a covariance map
    /// x: 2D array with a ToF trace in every row, requires the rows to sorted
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    /*
    #[pyfn(m)]
    fn pipico_equal_size<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        n_bins: usize,
        min: f64,
        max: f64,
    ) -> PyResult<&'py PyArray2<f64>> {
        let data = x.as_array();
        let n_part = data.shape()[1];
        let n_shots = data.shape()[0];

        // sort data into histogram iterating through data 2D array
        // initialize empty 2D histogram
        let mut hist = ndhistogram!(
            Uniform::<f64>::new(n_bins, min, max),
            Uniform::<f64>::new(n_bins, min, max)
        );
        let (mut p1, mut p2);
        for i in 0..n_shots {
            p1 = 0;
            while p1 < n_part {
                p2 = p1 + 1;
                while p2 < n_part {
                    hist.fill(&(data[[i, p1]], data[[i, p2]]));
                    p2 += 1;
                }
                p1 += 1;
            }
        }

        //let x = Array::<f64,_>::zeros((10, 10));
        //let b:Vec<f64> = hist.values().copied().collect();
        //println!("{:?}", b.len());
        //println!("{:?}", hist.axes());
        //// iterate the histogram values
        //for item in hist.iter() {
        //    println!("{:?}, {:?}, {:?}", item.index, item.bin, item.value)
        //}
        /*
        hist.values().enumerate().filter_map(|(i, v)| {
            let x = i % 12;
            let y = i / 12;
            if y > 0 && y < 11 && x > 0 && x < 11) { Some(v) }
            else { None }
        })
         */
        let a_hist: Array2<f64> = Array1::from_iter(hist.values().map(|v| *v).into_iter())
            .into_shape((n_bins + 2, n_bins + 2))
            .unwrap();

        Ok(a_hist
            .slice(s![1..n_bins + 1, 1..n_bins + 1])
            .to_pyarray(py))
    }
    */

    /// calculate a covariance map
    /// x: list of lists with a ToF trace in every row, pre-sorting not required
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    /*
    #[pyfn(m)]
    fn pipico_lists<'py>(
        py: Python<'py>,
        mut x: Vec<Vec<f64>>,
        n_bins: usize,
        min: f64,
        max: f64,
    ) -> PyResult<&'py PyArray2<f64>> {
        //let data = x.as_array();
        //let Nshots = data.shape()[0];
        //let mut data = x;

        // sort data in each row
        x.par_iter_mut()
            .for_each(|row| row.sort_by(|a, b| a.partial_cmp(b).unwrap()));

        // sort data into histogram iterating through data 2D array
        // initialize empty 2D histogram
        let mut hist = ndhistogram!(
            Uniform::<f64>::new(n_bins, min, max),
            Uniform::<f64>::new(n_bins, min, max)
        );

        let (mut p1, mut p2, mut n_part);
        for row in x {
            p1 = 0;
            n_part = row.len();
            while p1 < n_part {
                p2 = p1 + 1;
                while p2 < n_part {
                    hist.fill(&(row[p1], row[p2]));
                    p2 += 1;
                }
                p1 += 1;
            }
        }

        //let x = Array::<f64,_>::zeros((10, 10));
        //let b:Vec<f64> = hist.values().copied().collect();
        //println!("{:?}", b.len());
        //println!("{:?}", hist.axes());
        //// iterate the histogram values
        //for item in hist.iter() {
        //    println!("{:?}, {:?}, {:?}", item.index, item.bin, item.value)
        //}
        /*
        hist.values().enumerate().filter_map(|(i, v)| {
            let x = i % 12;
            let y = i / 12;
            if y > 0 && y < 11 && x > 0 && x < 11) { Some(v) }
            else { None }
        })
        // The .values() part seems to imply there are keys, and it's probably easier to iterate over entries and filter by keys...
         */
        let a_hist: Array2<f64> = Array1::from_iter(hist.values().map(|v| *v).into_iter())
            .into_shape((n_bins + 2, n_bins + 2))
            .unwrap();

        Ok(a_hist
            .slice(s![1..n_bins + 1, 1..n_bins + 1])
            .to_pyarray(py))

        //let dummy = Array::<f64,_>::zeros((10, 10));
        //Ok(dummy.into_pyarray(py))
    }
    */

    /// calculate a covariance map
    /// pydf: polars dataframe containing the data to compute
    /// col_grp: column name over which to perform the groupby
    /// col_pipico: column name over which the correlations should be calculated
    /// col_mask: column name which provides which should act to mask col_pipico
    /// Δr: width of the ring for the mask
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    /*
    #[pyfn(m)]
    fn polars_filter_momentum_pl<'py>(
        py: Python<'py>,
        pydf: PyDataFrame,
        col_grp: &str,
        col_pipico: &str,
        col_mask1: &str,
        col_mask2: &str,
        filter_delta: f64,
        n_bins: usize,
        hist_min: f64,
        hist_max: f64,
    //) -> PyResult<PyDataFrame> {
    ) -> PyResult<&'py PyArray2<f64>> {
        let df: DataFrame = pydf.into();

        // compute the covariances for a frame/row
        let compute_covariance = move |s: &mut [Series]| {

            // define 2D histogram into which the values get filled
            let mut hist = ndhistogram!(
                Uniform::<f64>::new(n_bins, hist_min, hist_max),
                Uniform::<f64>::new(n_bins, hist_min, hist_max)
            );

            let df = df!("col_cov"   => s[0].clone(),           // tof
                                    "col_mask1" => s[1].clone(),           // p_x
                                    "col_mask2" => s[2].clone()).unwrap(); // p_y
            let df = df.sort(["col_cov"], false).unwrap();

            // https://docs.rs/polars/0.26.1/polars/docs/eager/index.html#extracting-data
            let ca = df.column("col_cov").unwrap().f64().unwrap();
            let mut p2;
            for (p1, xx) in ca.into_iter().enumerate() {
                p2 = p1 + 1;
                let x = xx.unwrap();
                let filter1 = df.column("col_mask1").unwrap().f64().unwrap().get(p1).unwrap();
                let filter2 = df.column("col_mask2").unwrap().f64().unwrap().get(p1).unwrap();
                // filter data accordingly to the new x
                let masked = df
                    .clone()
                    .lazy()
                    .slice(p2 as i64, df.height() as u32)
                    .filter(lit(filter1).add(col("col_mask1")).pow(2).lt(filter_delta)
                       .and(lit(filter2).add(col("col_mask2")).pow(2).lt(filter_delta)) )
                    .select([col("col_cov")])
                    .collect()
                    .expect("Could filter col_mask");
                let masked_col = masked.column("col_cov").unwrap().f64().unwrap();
                for y in masked_col.into_iter() {
                    hist.fill(&(x, y.unwrap()));
                }
            }

            let a = hist.values().map(|v| *v).collect::<Vec<f64>>();
            Ok(Series::from_vec("hist", a))
        };

        // group df according to 'trigger nr' and calculate covariance for the frames
        let grouped = df
            .lazy()
            .groupby([col_grp])
            .agg([apply_multiple(
                compute_covariance,
                [col(col_pipico), col(col_mask1), col(col_mask2)],
                GetOutput::from_type(DataType::Float64),
                false,
            ).alias("histogram")])
            .select([col("histogram")])
            .collect()
            .expect("msg");
        let number_groups = grouped.height();

        // convert histogram we retrieve for every group into a single 2D histo
        let a = grouped.explode(["histogram"]).unwrap();
        let ca = a.column("histogram").unwrap().f64().unwrap();
        let to_vec = ca.into_iter().map(|f|
            match f {
                None => panic!("convert to vec: None"),
                Some(x) => x
            }).collect::<Vec<f64>>();
        let a_hist = Array1::from_iter(to_vec.into_iter())
            .into_shape((number_groups, (n_bins + 2).pow(2)))
            .unwrap()
            .sum_axis(Axis(0))
            .into_shape((n_bins + 2, n_bins + 2))
            .unwrap();

        Ok(a_hist
            .slice(s![1..n_bins + 1, 1..n_bins + 1])
            .to_pyarray(py))

        //Ok(PyDataFrame(df))
    }
    */

    /// calculate a covariance map
    /// pydf: polars dataframe containing the data to compute
    /// col_grp: column name over which to perform the groupby
    /// col_pipico: column name over which the correlations should be calculated
    /// col_mask: column name which provides which should act to mask col_pipico
    /// Δr: width of the ring for the mask
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    #[pyfn(m)]
    fn polars_filter_momentum_np<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        filter_delta: f64,
        n_bins: usize,
        hist_min: f64,
        hist_max: f64,
        //) -> PyResult<PyDataFrame> {
    ) -> PyResult<&'py PyArray2<f64>> {
        // x = [trigger nr, mz / tof, px, py] = 4 columns
        // need to get index in here as well, because I want to return the index of the pairs
        let data = x.as_array();
        let data_trigger = data.column(0);
        let data_tof = data.column(1);
        let data_px = data.column(2);
        let data_py = data.column(3);

        // define 2D histogram into which the values get filled
        let mut hist = ndhistogram!(
            Uniform::<f64>::new(n_bins, hist_min, hist_max),
            Uniform::<f64>::new(n_bins, hist_min, hist_max)
        );
        let trigger_nrs = data_trigger
            .iter()
            .map(|x| *x as i64)
            .unique()
            .collect_vec();
        let num_triggers = trigger_nrs.len();
        let num_cores = num_cpus::get() - 1;
        // makes the chunk to process n big, does not generate n chunks
        // chunk_iter = [[1,2,3], [4,5,6]]
        for chunk_iter in trigger_nrs.chunks(5) {
            //let mut data_chunk = Vec::<_>::with_capacity(1000);
            // TODO: check this to make this nicer: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#conversions-from-nested-vecsarrays
            // poor mans group-by along trigger nr (0 column)
            // check https://doc.rust-lang.org/nightly/core/iter/trait.Iterator.html#method.position maybe that's a better solution
            let chunk_vec = data
                .axis_iter(Axis(0))
                .into_iter()
                // maybe something like a 'df.query(`trigger nr` in @triggers)',
                // but as 'trigger nr' needs to be sorted in the first place it probably doesn't matter
                .filter(|x| {
                    (x[0] >= *chunk_iter.first().unwrap() as f64)
                        & (x[0] <= *chunk_iter.last().unwrap() as f64)
                })
                .flatten()
                .collect_vec();
            // convert the flattened groupby back into a 2D array with 4 columns
            let data_chunk = Array::from_shape_vec((chunk_vec.len() / 4, 4), chunk_vec).unwrap();
            // push this into a ThreadPool
            let trigger_nr = data_chunk
                .slice(s![.., 0])
                .iter()
                .map(|x| **x as i64)
                .unique()
                .collect_vec();
            for i in trigger_nr {
                // is it possible to only get the indices for the trigger nr in question and create a view on that slice?
                let trigger_frame_vec = data_chunk
                    .axis_iter(Axis(0))
                    .into_iter()
                    .filter(|x| *x[0] == i as f64)
                    .flatten()
                    .collect_vec();
                // same like above, is there a faster way?
                let trigger_frame =
                    Array::from_shape_vec((trigger_frame_vec.len() / 4, 4), trigger_frame_vec)
                        .unwrap();
                for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                    let p2 = p1 + 1;
                    let tof = *x[1];
                    let px = *x[2];
                    let py = *x[3];
                    let row = trigger_frame.slice(s![p2.., ..]);
                    let a = row
                        .axis_iter(Axis(0))
                        .into_iter()
                        .filter(|&x| ((*x[2] + *px).powf(2.) < 0.01) & ((*x[3] + *py).powf(2.) < 0.01))
                        .map(|x| x[1])
                        .collect_vec();
                    for y in a {
                        hist.fill(&(*tof, **y));
                        // collect pairs
                    }
                }
            }
            //let b = Array::from_vec(a.fl);
            //dbg!();
        }

        let a_hist: Array2<f64> = Array1::from_iter(hist.values().map(|v| *v).into_iter())
            .into_shape((n_bins + 2, n_bins + 2))
            .unwrap();

        Ok(a_hist
            .slice(s![1..n_bins + 1, 1..n_bins + 1])
            .to_pyarray(py))
    }

    #[pyfn(m)]
    fn hallo<'py>(py: Python<'py>, num: i64) {
        println!("hallo {:?}", num);
    }

    Ok(())
}

/// generate array of random numbers and check if none are double with reference list
///
pub fn get_bg_idx(rng: &mut ThreadRng) {
    let distribution = rand::distributions::Uniform::new(0, 1_000);
    let ref_data = (100..200).into_iter().collect_vec();
    dbg!(ref_data);

    let mut idx_bg = (rng)
        .sample_iter(distribution)
        .take(100)
        .collect::<Vec<_>>();
    /*
    let idx_bg = (0..100)
        .into_iter()
        .map(|| {
            while let x = rng.gen_range(0..1_000);

        });
     */
}

// data needs to be sorted along triggers
// here we use index on slices
pub fn polars_filter_momentum_bench_idx(
    data: Array2<f64>,
    //) -> PyResult<PyDataFrame> {
) -> Array2<f64> {
    let filter_delta = 0.01;
    let n_bins = 10;
    let hist_min = 0.;
    let hist_max = 10.;

    let data_trigger = data.column(0);
    let data_tof = data.column(1);
    let data_px = data.column(2);
    let data_py = data.column(3);
    // define 2D histogram into which the values get filled
    let mut hist = ndhistogram!(
        Uniform::<f64>::new(n_bins, hist_min, hist_max),
        Uniform::<f64>::new(n_bins, hist_min, hist_max)
    );
    //let trigger_nrs = data.slice(s![..,0]).iter().map(|x| *x as i64).unique().collect_vec();
    let trigger_nrs = data_trigger
        .iter()
        .map(|x| *x as i64)
        .unique()
        .collect_vec();
    let num_triggers = trigger_nrs.len();
    let num_cores = num_cpus::get() - 1;

    // iterate over chunks, the computation of a chunk should be pushed into a thread
    // chunks are defined as group of triggers, the size is determined by the number of CPU cores
    // chunksize determines the size of the chunk, so if we want to unload all data evenly onto the cores
    // we need to do `num_trigger / num_cores`
    // `chunk_triggers` will contain the trigger number which belong to a chunk
    for chunk_triggers in trigger_nrs.chunks(num_triggers / num_cores) {
        //let mut data_chunk = Vec::<_>::with_capacity(1000);
        // TODO: check this to make this nicer: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#conversions-from-nested-vecsarrays
        // collect all data which belong to a chunk in a "DataFrame"
        let chunk_vec = data
            .axis_iter(Axis(0))
            .into_iter()
            .filter(|x| {
                (x[0] >= *chunk_triggers.first().unwrap() as f64)
                    & (x[0] <= *chunk_triggers.last().unwrap() as f64)
            })
            .flatten()
            .collect_vec();
        // collect all indices which belong to one chunk
        let data_chunk =
            Array::from_shape_vec((chunk_vec.len() / data.ncols(), data.ncols()), chunk_vec)
                .unwrap();

        // inititalise random number generator
        let mut rng = rand::thread_rng();
        let distribution = rand::distributions::Uniform::new(0, data_trigger.len());

        // push this into a ThreadPool
        //let trigger_nr = data_chunk.slice(s![..,0]).iter().map(|x| **x as i64).unique().collect_vec();
        //dbg!(trigger_nr);
        for trg_nr in chunk_triggers {
            // get only indices which belong to a group, something like
            // `df.query('trigger in @chunk_triggers')`
            // provides the index of the trigger numbers which belong in this chunk
            // didn't work, because I cannot access the arrayview providing an list of indices
            // because the trigger numbers are sorted, the index of the trigger numbers should be increasing
            // and such something like s![idx[0]..idx[-1]] should work

            // get the index for the current group
            let idx_grp = data_trigger
                .iter()
                .positions(|v| *v as i64 == *trg_nr)
                .collect::<Vec<_>>();
            // get the index for a background from a random sample
            /*
            let mut idx_bg = (&mut rng)
                .sample_iter(distribution)
                .take(idx_grp.len())
                .collect::<Vec<_>>();
             */
            //dbg!(idx_bg);

            for (p1, idx) in idx_grp.iter().enumerate() {
                let p2 = p1 + 1;
                let tof_x = data_tof[p1];
                let px = data_px[p1];
                let py = data_py[p1];

                let row = izip!(data_tof.slice(s![p2..]), data_px.slice(s![p2..]), data_py.slice(s![p2..]))
                    .into_iter()
                    .filter(|&(_, p2_px, p2_py)| ((*p2_px + px).powf(2.) + (*p2_py + py).powf(2.)) < (px * px + py * py) * 0.00025)
                    .map(|(&p2_tof, _, _)| p2_tof)
                    .collect_vec();
                for tof_y in row {
                    hist.fill(&(tof_x, tof_y));
                }
            }
        }
        //let b = Array::from_vec(a.fl);
        //dbg!();
    }

    let a_hist: Array2<f64> = Array1::from_iter(hist.values().map(|v| *v).into_iter())
        .into_shape((n_bins + 2, n_bins + 2))
        .unwrap();

    a_hist.slice(s![1..n_bins + 1, 1..n_bins + 1]).to_owned()
}

// data needs to be sorted along triggers
// this should already work
// works with 2D array
pub fn polars_filter_momentum_bench_2D(
    data: Array2<f64>,
    //) -> PyResult<PyDataFrame> {
) -> Array2<f64> {
    let filter_delta = 0.01;
    let n_bins = 10;
    let hist_min = 0.;
    let hist_max = 10.;

    let data_trigger = data.column(0);
    let data_tof = data.column(1);
    let data_px = data.column(2);
    let data_py = data.column(3);
    // define 2D histogram into which the values get filled
    let mut hist = ndhistogram!(
        Uniform::<f64>::new(n_bins, hist_min, hist_max),
        Uniform::<f64>::new(n_bins, hist_min, hist_max)
    );
    //let trigger_nrs = data.slice(s![..,0]).iter().map(|x| *x as i64).unique().collect_vec();
    let trigger_nrs = data_trigger
        .iter()
        .map(|x| *x as i64)
        .unique()
        .collect_vec();
    let num_triggers = trigger_nrs.len();
    let num_cores = num_cpus::get() - 1;

    // iterate over chunks, the computation of a chunk should be pushed into a thread
    // chunks are defined as group of triggers, the size is determined by the number of CPU cores
    // chunksize determines the size of the chunk, so if we want to unload all data evenly onto the cores
    // we need to do `num_trigger / num_cores`
    // `chunk_triggers` will contain the trigger number which belong to a chunk
    for chunk_triggers in trigger_nrs.chunks(num_triggers / num_cores) {
        //let mut data_chunk = Vec::<_>::with_capacity(1000);
        // TODO: check this to make this nicer: https://docs.rs/ndarray/latest/ndarray/struct.ArrayBase.html#conversions-from-nested-vecsarrays
        // collect all data which belong to a chunk in a "DataFrame"
        let chunk_vec = data
            .axis_iter(Axis(0))
            .into_iter()
            .filter(|x| {
                (x[0] >= *chunk_triggers.first().unwrap() as f64)
                    & (x[0] <= *chunk_triggers.last().unwrap() as f64)
            })
            .flatten()
            .collect_vec();
        // collect all indices which belong to one chunk
        let data_chunk =
            Array::from_shape_vec((chunk_vec.len() / data.ncols(), data.ncols()), chunk_vec)
                .unwrap();

        // inititalise random number generator
        let mut rng = rand::thread_rng();
        let distribution = rand::distributions::Uniform::new(0, data_trigger.len());

        // push this into a ThreadPool
        /*
        let trigger_nr = data_chunk
            .slice(s![.., 0])
            .iter()
            .map(|x| **x as i64)
            .unique()
            .collect_vec();
         */
        //dbg!(trigger_nr);
        for trg_nr in chunk_triggers {
            let trigger_frame_vec = data_chunk
                .axis_iter(Axis(0))
                .into_iter()
                .filter(|x| *x[0] == *trg_nr as f64)
                .flatten()
                .collect_vec();
            let trigger_frame = Array::from_shape_vec(
                (trigger_frame_vec.len() / data.ncols(), data.ncols()),
                trigger_frame_vec,
            )
            .unwrap();

            for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                let p2 = p1 + 1;
                let tof_x = *x[1];
                let px = *x[2];
                let py = *x[3];

                let row = trigger_frame.slice(s![p2.., ..]);
                let a = row
                    .axis_iter(Axis(0))
                    .into_iter()
                    .filter(|&x| ((*x[2] + *px).powf(2.) + (*x[3] + *py).powf(2.)) < (*px * *px + *py * *py) * 0.00025)
                    .map(|x| x[1])
                    .collect_vec();

                for tof_y in a {
                    hist.fill(&(*tof_x, **tof_y));
                }
            }
        }
        //let b = Array::from_vec(a.fl);
        //dbg!();
    }

    let a_hist: Array2<f64> = Array1::from_iter(hist.values().map(|v| *v).into_iter())
        .into_shape((n_bins + 2, n_bins + 2))
        .unwrap();

    a_hist.slice(s![1..n_bins + 1, 1..n_bins + 1]).to_owned()
}
