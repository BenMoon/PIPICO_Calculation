//#![feature(iter_collect_into)]

use rand::rngs::ThreadRng;
use rand::{self, Rng};
use std::collections::HashSet;
use std::usize;

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{s, Array1, Array2, ViewRepr};
use ndhistogram::{axis::Uniform, ndhistogram, Histogram};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
extern crate num_cpus;

use rayon::prelude::*;

/// calculate a covariance map
#[pymodule]
fn pipico(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// toy code remaining for demonstration purposes
    /// calculate a covariance map
    /// x: list of lists with a ToF trace in every row, pre-sorting not required
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
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

    /// deprecated
    /// calculate a covariance map, directly returning the histogram, only considering px and py, no background subtraction
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
            let data_chunk =
                Array::from_shape_vec((chunk_vec.len() / data.ncols(), data.ncols()), chunk_vec)
                    .unwrap();
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
                let trigger_frame = Array::from_shape_vec(
                    (trigger_frame_vec.len() / data.ncols(), data.ncols()),
                    trigger_frame_vec,
                )
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
                        .filter(|&x| {
                            ((*x[2] + *px).powf(2.) + (*x[3] + *py).powf(2.))
                                < (*px * *px + *py * *py) * filter_delta
                        })
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

    /// Extract ion pairs fulfilling the condition of a 3D momentum conservation
    /// x: 2D array containing the data [trigger nr, index, px, py, pz, tof]
    /// momentum_cut: realtiv cut on Newton sphere
    #[pyfn(m)]
    fn get_covar_pairs<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        momentum_cut: f64,
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
        let data = x.as_array();
        let data_trigger = data.column(0);

        // vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
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
        // chunk_triggers = (142) &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
        //for chunk_triggers in trigger_nrs.chunks(num_triggers / num_cores) {
        let all_pairs = trigger_nrs
            .chunks(num_triggers / num_cores)
            .map(|chunk_triggers| {
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
                // collect all data which belong to one chunk
                let data_chunk = Array::from_shape_vec(
                    (chunk_vec.len() / data.ncols(), data.ncols()),
                    chunk_vec,
                )
                .unwrap();

                // https://faraday.ai/blog/saved-by-the-compiler-parallelizing-a-loop-with-rust-and-rayon
                let chunk_pairs = chunk_triggers
                    .par_iter()
                    .map(|trg_nr| {
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

                        /* calculate covariance */
                        let mut pairs = Vec::with_capacity(trigger_frame.nrows() / 5);
                        for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                            let p2 = p1 + 1;
                            let idx_x = *x[1];
                            let px = *x[2];
                            let py = *x[3];
                            let pz = *x[4];

                            let row = trigger_frame.slice(s![p2.., ..]);
                            let a = row
                                .axis_iter(Axis(0))
                                .into_iter()
                                .filter(|&x| {
                                    ((*x[2] + *px).powf(2.)
                                        + (*x[3] + *py).powf(2.)
                                        + (*x[4] + *pz).powf(2.))
                                        <= (*px * *px + *py * *py + *pz * *pz) * momentum_cut
                                })
                                .map(|x| x[1])
                                .collect_vec();

                            for idx_y in a {
                                pairs.push([*idx_x, **idx_y]);
                            }
                        }

                        /* calculate the background */
                        // inititalise random number generator
                        let mut rng = rand::thread_rng();
                        let trg_frame_indizes = trigger_frame.slice(s![.., 1]);
                        let bg_frame_idx = get_bg_idx(&mut rng, trg_frame_indizes, data.nrows());
                        let bg_frame = data.select(Axis(0), &bg_frame_idx);
                        let mut pairs_bg = Vec::with_capacity(trigger_frame.nrows() / 5);
                        for (p1, x) in bg_frame.axis_iter(Axis(0)).enumerate() {
                            let p2 = p1 + 1;
                            let idx_x = x[1];
                            let px = x[2];
                            let py = x[3];
                            let pz = x[4];
                            let tof = x[5];

                            let row = trigger_frame.slice(s![p2.., ..]);
                            let a = row
                                .axis_iter(Axis(0))
                                .into_iter()
                                .filter(|&x| {
                                    ((*x[2] + px).powf(2.)
                                        + (*x[3] + py).powf(2.)
                                        + (*x[4] + pz).powf(2.))
                                        <= (px * px + py * py + pz * pz) * momentum_cut
                                })
                                .map(|x| (x[1], x[5]))
                                .collect_vec();

                            for (idx_y, tof_y) in a {
                                if tof <= **tof_y {
                                    pairs_bg.push([idx_x, **idx_y]);
                                } else {
                                    pairs_bg.push([**idx_y, idx_x]);
                                }
                            }
                        }

                        (pairs, pairs_bg)
                    })
                    .collect::<Vec<_>>();

                let mut fg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
                let mut bg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
                for i in chunk_pairs.into_iter() {
                    for j in i.0 {
                        fg.push(j);
                    }
                    for j in i.1 {
                        bg.push(j)
                    }
                }

                (fg, bg)
            })
            .collect::<Vec<_>>();

        let mut fg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 5); // 1/5 is data, not sure how good this guess is
        let mut bg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 10); // 1/10 is bg
        for i in all_pairs.into_iter() {
            for j in i.0 {
                fg.push(j);
            }
            for j in i.1 {
                bg.push(j)
            }
        }

        let a = Array2::from(fg).to_pyarray(py);
        let b = Array2::from(bg).to_pyarray(py);
        Ok((a, b))
    }

    /// Extract ion pairs fulfilling the condition of a 3D momentum conservation
    /// x: 2D array containing the data [trigger nr, index, px, py, pz, tof, mass]
    /// mass_momentum_cut: px+py+pz <= mass_momentum_cut for masses define in this array, e.g.
    /// ```
    ///     mass_momementum_cut = np.array(
    ///         [[17.5, 18.5, 17.5, 18.5, 1e1**2],
    ///         [16.5, 17.5, 17.5, 18.5, 2.9e1**2],
    ///         [16.5, 17.5, 18.5, 19.5, 9.7e1**2]])
    ///     default_momentum_cut = 3e5**2
    /// ```
    /// default_momentum_cut: momentum sphere which applies for all other masses
    #[pyfn(m)]
    fn get_covar_pairs_fixed_cut<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        mass_momentum_cut: PyReadonlyArray2<'py, f64>,
        default_momentum_cut: f64,
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray2<f64>)> {
        let data = x.as_array();
        let mp_cut = mass_momentum_cut.as_array();

        let data_trigger = data.column(0);

        //let trigger_nrs = data.slice(s![..,0]).iter().map(|x| *x as i64).unique().collect_vec();
        // vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
        let trigger_nrs = data_trigger
            .iter()
            .map(|x| *x as i64)
            .unique()
            .collect_vec();
        let num_triggers = trigger_nrs.len();
        let num_cores = 1; //num_cpus::get() - 1;

        // iterate over chunks, the computation of a chunk should be pushed into a thread
        // chunks are defined as group of triggers, the size is determined by the number of CPU cores
        // chunksize determines the size of the chunk, so if we want to unload all data evenly onto the cores
        // we need to do `num_trigger / num_cores`
        // `chunk_triggers` will contain the trigger number which belong to a chunk
        // chunk_triggers = (142) &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
        //for chunk_triggers in trigger_nrs.chunks(num_triggers / num_cores) {
        let all_pairs = trigger_nrs
            .chunks(num_triggers / num_cores)
            .map(|chunk_triggers| {
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
                // collect all data which belong to one chunk
                let data_chunk = Array::from_shape_vec(
                    (chunk_vec.len() / data.ncols(), data.ncols()),
                    chunk_vec,
                )
                .unwrap();

                // https://faraday.ai/blog/saved-by-the-compiler-parallelizing-a-loop-with-rust-and-rayon
                let chunk_pairs = chunk_triggers
                    .par_iter()
                    .map(|trg_nr| {
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

                        /* calculate covariance */
                        let mut pairs = Vec::with_capacity(trigger_frame.nrows() / 5);
                        for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                            let p2 = p1 + 1;
                            let idx_x = *x[1];
                            let px = *x[2];
                            let py = *x[3];
                            let pz = *x[4];
                            let m1 = *x[6];

                            let row = trigger_frame.slice(s![p2.., ..]);
                            let a = row
                                .axis_iter(Axis(0))
                                .into_iter()
                                .filter(|&x| {
                                    let p_sum = ((*x[2] + *px).powf(2.)
                                        + (*x[3] + *py).powf(2.)
                                        + (*x[4] + *pz).powf(2.));
                                    let m2 = *x[6];

                                    // apply for mass specific momentum cut
                                    for i in (0..mp_cut.nrows()).into_iter() {
                                        if m1 >= &mp_cut[[i, 0]]
                                            && m1 < &mp_cut[[i, 1]]
                                            && m2 >= &mp_cut[[i, 2]]
                                            && m2 < &mp_cut[[i, 3]]
                                        {
                                            if p_sum <= mp_cut[[i, 4]] {
                                                return true;
                                            } else {
                                                return false;
                                            }
                                        }
                                    }
                                    // apply for all other chases
                                    if p_sum <= default_momentum_cut {
                                        return true;
                                    }
                                    false
                                })
                                .map(|x| x[1])
                                .collect_vec();

                            for idx_y in a {
                                pairs.push([*idx_x, **idx_y]);
                            }
                        }

                        /* calculate the background */
                        // inititalise random number generator
                        let mut rng = rand::thread_rng();
                        let trg_frame_indizes = trigger_frame.slice(s![.., 1]);
                        let bg_frame_idx = get_bg_idx(&mut rng, trg_frame_indizes, data.nrows());
                        let bg_frame = data.select(Axis(0), &bg_frame_idx);
                        let mut pairs_bg = Vec::with_capacity(trigger_frame.nrows() / 5);
                        for (p1, x) in bg_frame.axis_iter(Axis(0)).enumerate() {
                            let p2 = p1 + 1;
                            let idx_x = x[1];
                            let px = x[2];
                            let py = x[3];
                            let pz = x[4];
                            let tof = x[5];
                            let m1 = x[6];

                            let row = trigger_frame.slice(s![p2.., ..]);
                            let a = row
                                .axis_iter(Axis(0))
                                .into_iter()
                                .filter(|&x| {
                                    let p_sum = ((*x[2] + px).powf(2.)
                                        + (*x[3] + py).powf(2.)
                                        + (*x[4] + pz).powf(2.));
                                    let m2 = *x[6];

                                    // apply for mass specific momentum cut
                                    for i in (0..mp_cut.nrows()).into_iter() {
                                        if m1 >= mp_cut[[i, 0]]
                                            && m1 < mp_cut[[i, 1]]
                                            && m2 >= &mp_cut[[i, 2]]
                                            && m2 < &mp_cut[[i, 3]]
                                        {
                                            if p_sum <= mp_cut[[i, 4]] {
                                                return true;
                                            } else {
                                                return false;
                                            }
                                        }
                                    }
                                    // apply for all other chases
                                    if p_sum <= default_momentum_cut {
                                        return true;
                                    }
                                    false
                                })
                                .map(|x| (x[1], x[5]))
                                .collect_vec();

                            for (idx_y, tof_y) in a {
                                if tof <= **tof_y {
                                    pairs_bg.push([idx_x, **idx_y]);
                                } else {
                                    pairs_bg.push([**idx_y, idx_x]);
                                }
                            }
                        }

                        (pairs, pairs_bg)
                    })
                    .collect::<Vec<_>>();

                let mut fg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
                let mut bg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
                for i in chunk_pairs.into_iter() {
                    for j in i.0 {
                        fg.push(j);
                    }
                    for j in i.1 {
                        bg.push(j)
                    }
                }

                //chunk_pairs.into_iter().flatten().collect::<Vec<_>>()
                (fg, bg)
                //vec![1., 2., 3.]
            })
            .collect::<Vec<_>>();

        //let b = all_pairs.into_iter().flatten().collect::<Vec<_>>();

        let mut fg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 5); // 1/5 is data, not sure how good this guess is
        let mut bg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 10); // 1/10 is bg
        for i in all_pairs.into_iter() {
            for j in i.0 {
                fg.push(j);
            }
            for j in i.1 {
                bg.push(j)
            }
        }

        let a = Array2::from(fg).to_pyarray(py);
        let b = Array2::from(bg).to_pyarray(py);
        Ok((a, b))

        //let a_hist = Array1::from_vec(vec![1.,2.,3.,4.]).into_shape((2,2)).unwrap();
        //Ok((a_hist
        //    .to_pyarray(py)))
    }

    /// generate array of random numbers and check if none are double with reference list
    fn get_bg_idx(
        rng: &mut ThreadRng,
        idx_trg_frame: ArrayBase<ViewRepr<&&&f64>, Dim<[usize; 1]>>,
        max_index: usize,
    ) -> Vec<usize> {
        let idx_set: HashSet<usize> =
            HashSet::from_iter(idx_trg_frame.into_iter().map(|x| ***x as usize));
        let mut idx_bg = HashSet::with_capacity(idx_trg_frame.len());
        let mut bg: usize;

        for _ in (0..idx_trg_frame.len()).into_iter() {
            loop {
                bg = rng.gen_range(0..max_index);
                if !idx_set.contains(&bg) && !idx_bg.contains(&bg) {
                    idx_bg.insert(bg);
                    break;
                }
            }
        }

        idx_bg.into_iter().collect_vec()
    }

    Ok(())
}

/// rust implementation for benchmarking
/// generate array of random numbers and check if none are double with reference list
pub fn get_bg_idx(
    rng: &mut ThreadRng,
    idx_trg_frame: ArrayBase<ViewRepr<&&&f64>, Dim<[usize; 1]>>,
    max_index: usize,
) -> Vec<usize> {
    let mut idx_bg = Vec::<usize>::with_capacity(idx_trg_frame.len());
    let mut bg: usize;

    for _ in (0..idx_trg_frame.len()).into_iter() {
        loop {
            bg = rng.gen_range(0..max_index);
            if idx_trg_frame.iter().all(|&x| **x != bg as f64) && idx_bg.iter().all(|&x| x != bg) {
                idx_bg.push(bg);
                break;
            }
        }
    }
    idx_bg
}

/// rust implementation for benchmarking
/// generate array of random numbers and check if none are double with reference list
pub fn get_bg_idx_set(
    rng: &mut ThreadRng,
    idx_trg_frame: ArrayBase<ViewRepr<&&&f64>, Dim<[usize; 1]>>,
    max_index: usize,
) -> Vec<usize> {
    let idx_set: HashSet<usize> =
        HashSet::from_iter(idx_trg_frame.into_iter().map(|x| ***x as usize));
    let mut idx_bg = HashSet::with_capacity(idx_trg_frame.len());
    let mut bg: usize;

    for _ in (0..idx_trg_frame.len()).into_iter() {
        loop {
            bg = rng.gen_range(0..max_index);
            if !idx_set.contains(&bg) && !idx_bg.contains(&bg) {
                idx_bg.insert(bg);
                break;
            }
        }
    }

    idx_bg.into_iter().collect_vec()
}

/// rust implementation for benchmarking
// data needs to be sorted along triggers
// this should already work
// works with 2D array
pub fn ndarray_filter_momentum_bench_2D(
    data: Array2<f64>,
    //) -> PyResult<PyDataFrame> {
) -> Array2<f64> {
    let filter_delta = 0.01;
    let n_bins = 100;
    let hist_min = 0.;
    let hist_max = 10.;

    let data_trigger = data.column(0);

    //let trigger_nrs = data.slice(s![..,0]).iter().map(|x| *x as i64).unique().collect_vec();
    // vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
    let trigger_nrs = data_trigger
        .iter()
        .map(|x| *x as i64)
        .unique()
        .collect_vec();
    let num_triggers = trigger_nrs.len();
    let num_cores = num_cpus::get() - 1;

    let cov_hist = trigger_nrs
        .chunks(num_triggers / num_cores)
        .map(|chunk_triggers| {
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
            // collect all data which belong to one chunk
            let data_chunk =
                Array::from_shape_vec((chunk_vec.len() / data.ncols(), data.ncols()), chunk_vec)
                    .unwrap();

            // https://faraday.ai/blog/saved-by-the-compiler-parallelizing-a-loop-with-rust-and-rayon
            let chunk_hists = chunk_triggers
                .par_iter()
                .map(|trg_nr| {
                    // define 2D histogram into which the values get filled
                    let mut hist = ndhistogram!(
                        Uniform::<f64>::new(n_bins, hist_min, hist_max),
                        Uniform::<f64>::new(n_bins, hist_min, hist_max)
                    );
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

                    /* calculate covariance */
                    for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                        let p2 = p1 + 1;
                        let tof_x = *x[1];
                        let px = *x[2];
                        let py = *x[3];

                        let row = trigger_frame.slice(s![p2.., ..]);
                        let a = row
                            .axis_iter(Axis(0))
                            .into_iter()
                            .filter(|&x| {
                                ((*x[2] + *px).powf(2.) + (*x[3] + *py).powf(2.))
                                    < (*px * *px + *py * *py) * 0.0025
                            })
                            .map(|x| x[1])
                            .collect_vec();

                        for tof_y in a {
                            hist.fill(&(*tof_x, **tof_y));
                        }
                    }

                    hist
                })
                .reduce_with(|hists, hist| (hists + &hist).expect("Axes are compatible"))
                .unwrap();

            chunk_hists
        })
        .reduce(|hists, hist| (hists + &hist).expect("Axes are compatible"))
        .unwrap();

    let a_hist: Array2<f64> = Array1::from_iter(cov_hist.values().map(|v| *v).into_iter())
        .into_shape((n_bins + 2, n_bins + 2))
        .unwrap();

    a_hist.slice(s![1..n_bins + 1, 1..n_bins + 1]).to_owned()
}

/// rust implementation for benchmarking
pub fn get_pairs_bench(data: Array2<f64>) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let filter_delta = 0.01;
    let n_bins = 10;
    let hist_min = 0.;
    let hist_max = 10.;

    let data_trigger = data.column(0);

    //let trigger_nrs = data.slice(s![..,0]).iter().map(|x| *x as i64).unique().collect_vec();
    // vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
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
    // chunk_triggers = (142) &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
    //for chunk_triggers in trigger_nrs.chunks(num_triggers / num_cores) {
    let all_pairs = trigger_nrs
        .chunks(num_triggers / num_cores)
        .map(|chunk_triggers| {
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
            // collect all data which belong to one chunk
            let data_chunk =
                Array::from_shape_vec((chunk_vec.len() / data.ncols(), data.ncols()), chunk_vec)
                    .unwrap();

            // https://faraday.ai/blog/saved-by-the-compiler-parallelizing-a-loop-with-rust-and-rayon
            let chunk_pairs = chunk_triggers
                .par_iter()
                .map(|trg_nr| {
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

                    /* calculate covariance */
                    let mut pairs = Vec::with_capacity(trigger_frame.nrows() / 5);
                    for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                        let p2 = p1 + 1;
                        let idx_x = *x[1];
                        let px = *x[2];
                        let py = *x[3];
                        let pz = *x[4];

                        let row = trigger_frame.slice(s![p2.., ..]);
                        let a = row
                            .axis_iter(Axis(0))
                            .into_iter()
                            .filter(|&x| {
                                ((*x[2] + *px).powf(2.)
                                    + (*x[3] + *py).powf(2.)
                                    + (*x[4] + *pz).powf(2.))
                                    < (*px * *px + *py * *py + *pz * *pz) * 0.0025
                            })
                            .map(|x| x[1])
                            .collect_vec();

                        for idx_y in a {
                            pairs.push([*idx_x, **idx_y]);
                        }
                    }

                    /* calculate the background */
                    // inititalise random number generator
                    let mut rng = rand::thread_rng();
                    let trg_frame_indizes = trigger_frame.slice(s![.., 1]);
                    //let bg_frame_idx = get_bg_idx(&mut rng, trg_frame_indizes, data.nrows());
                    let bg_frame_idx = get_bg_idx_set(&mut rng, trg_frame_indizes, data.nrows());
                    let bg_frame = data.select(Axis(0), &bg_frame_idx);
                    let mut pairs_bg = Vec::with_capacity(trigger_frame.nrows() / 5);
                    for (p1, x) in bg_frame.axis_iter(Axis(0)).enumerate() {
                        let p2 = p1 + 1;
                        let idx_x = x[1];
                        let px = x[2];
                        let py = x[3];
                        let pz = x[4];

                        let row = trigger_frame.slice(s![p2.., ..]);
                        let a = row
                            .axis_iter(Axis(0))
                            .into_iter()
                            .filter(|&x| {
                                ((*x[2] + px).powf(2.)
                                    + (*x[3] + py).powf(2.)
                                    + (*x[4] + pz).powf(2.))
                                    < (px * px + py * py + pz * pz) * 0.0025
                            })
                            .map(|x| x[1])
                            .collect_vec();

                        for idx_y in a {
                            pairs_bg.push([idx_x, **idx_y]);
                        }
                    }

                    (pairs, pairs_bg)
                })
                .collect::<Vec<_>>();

            let mut fg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
            let mut bg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
            for i in chunk_pairs.into_iter() {
                for j in i.0 {
                    fg.push(j);
                }
                for j in i.1 {
                    bg.push(j)
                }
            }

            (fg, bg)
        })
        .collect::<Vec<_>>();

    let mut fg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 5); // 1/5 is data
    let mut bg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 10); // 1/10 is bg
    for i in all_pairs.into_iter() {
        for j in i.0 {
            fg.push(j);
        }
        for j in i.1 {
            bg.push(j)
        }
    }

    (fg, bg)
}

/// rust implementation for benchmarking
pub fn get_covar_pairs_fixed_cut(
    x: Array2<f64>,
    mass_momentum_cut: Array2<f64>,
    default_momentum_cut: f64,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let data = x;
    let mp_cut = mass_momentum_cut;

    let data_trigger = data.column(0);

    //let trigger_nrs = data.slice(s![..,0]).iter().map(|x| *x as i64).unique().collect_vec();
    // vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
    let trigger_nrs = data_trigger
        .iter()
        .map(|x| *x as i64)
        .unique()
        .collect_vec();
    let num_triggers = trigger_nrs.len();
    let num_cores = 1; //num_cpus::get() - 1;

    // iterate over chunks, the computation of a chunk should be pushed into a thread
    // chunks are defined as group of triggers, the size is determined by the number of CPU cores
    // chunksize determines the size of the chunk, so if we want to unload all data evenly onto the cores
    // we need to do `num_trigger / num_cores`
    // `chunk_triggers` will contain the trigger number which belong to a chunk
    // chunk_triggers = (142) &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
    //for chunk_triggers in trigger_nrs.chunks(num_triggers / num_cores) {
    let all_pairs = trigger_nrs
        .chunks(num_triggers / num_cores)
        .map(|chunk_triggers| {
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
            // collect all data which belong to one chunk
            let data_chunk =
                Array::from_shape_vec((chunk_vec.len() / data.ncols(), data.ncols()), chunk_vec)
                    .unwrap();

            // https://faraday.ai/blog/saved-by-the-compiler-parallelizing-a-loop-with-rust-and-rayon
            let chunk_pairs = chunk_triggers
                .iter() //par_iter
                .map(|trg_nr| {
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

                    /* calculate covariance */
                    let mut pairs = Vec::with_capacity(trigger_frame.nrows() / 5);
                    for (p1, x) in trigger_frame.axis_iter(Axis(0)).enumerate() {
                        let p2 = p1 + 1;
                        let idx_x = *x[1];
                        let px = *x[2];
                        let py = *x[3];
                        let pz = *x[4];
                        let m1 = *x[6];

                        let row = trigger_frame.slice(s![p2.., ..]);
                        let a = row
                            .axis_iter(Axis(0))
                            .into_iter()
                            .filter(|&x| {
                                let p_sum = (*x[2] + *px).powf(2.)
                                    + (*x[3] + *py).powf(2.)
                                    + (*x[4] + *pz).powf(2.);
                                let m2 = *x[6];

                                // apply for mass specific momentum cut
                                for i in (0..mp_cut.nrows()).into_iter() {
                                    if m1 >= &mp_cut[[i, 0]]
                                        && m1 < &mp_cut[[i, 1]]
                                        && m2 >= &mp_cut[[i, 2]]
                                        && m2 < &mp_cut[[i, 3]]
                                    {
                                        if p_sum <= mp_cut[[i, 4]] {
                                            return true;
                                        } else {
                                            return false;
                                        }
                                    }
                                }
                                // apply for all other chases
                                if p_sum <= default_momentum_cut {
                                    return true;
                                }
                                false
                            })
                            .map(|x| x[1])
                            .collect_vec();

                        for idx_y in a {
                            dbg!(idx_x, &idx_y);
                            pairs.push([*idx_x, **idx_y]);
                        }
                    }

                    /* calculate the background */
                    // inititalise random number generator
                    let mut rng = rand::thread_rng();
                    let trg_frame_indizes = trigger_frame.slice(s![.., 1]);
                    dbg!(&trg_frame_indizes);
                    let bg_frame_idx = get_bg_idx(&mut rng, trg_frame_indizes, data.nrows());
                    let bg_frame = data.select(Axis(0), &bg_frame_idx);
                    let mut pairs_bg = Vec::with_capacity(trigger_frame.nrows() / 5);
                    for (p1, x) in bg_frame.axis_iter(Axis(0)).enumerate() {
                        let p2 = p1 + 1;
                        let idx_x = x[1];
                        let px = x[2];
                        let py = x[3];
                        let pz = x[4];
                        let tof = x[5];
                        let m1 = x[6];

                        let row = trigger_frame.slice(s![p2.., ..]);
                        let a = row
                            .axis_iter(Axis(0))
                            .into_iter()
                            .filter(|&x| {
                                let p_sum = ((*x[2] + px).powf(2.)
                                    + (*x[3] + py).powf(2.)
                                    + (*x[4] + pz).powf(2.));
                                let m2 = *x[6];

                                // apply for mass specific momentum cut
                                for i in (0..mp_cut.nrows()).into_iter() {
                                    if m1 >= mp_cut[[i, 0]]
                                        && m1 < mp_cut[[i, 1]]
                                        && m2 >= &mp_cut[[i, 2]]
                                        && m2 < &mp_cut[[i, 3]]
                                    {
                                        if p_sum <= mp_cut[[i, 4]] {
                                            return true;
                                        } else {
                                            return false;
                                        };
                                    }
                                }
                                // apply for all other chases
                                if p_sum <= default_momentum_cut {
                                    return true;
                                }
                                false
                            })
                            .map(|x| (x[1], x[5]))
                            .collect_vec();

                        for (idx_y, tof_y) in a {
                            if tof <= **tof_y {
                                pairs_bg.push([idx_x, **idx_y]);
                            } else {
                                pairs_bg.push([**idx_y, idx_x]);
                            }
                        }
                    }

                    (pairs, pairs_bg)
                })
                .collect::<Vec<_>>();

            let mut fg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
            let mut bg = Vec::<[f64; 2]>::with_capacity(chunk_pairs.len());
            for i in chunk_pairs.into_iter() {
                for j in i.0 {
                    fg.push(j);
                }
                for j in i.1 {
                    bg.push(j)
                }
            }

            (fg, bg)
        })
        .collect::<Vec<_>>();

    let mut fg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 5); // 1/5 is data, not sure how good this guess is
    let mut bg = Vec::<[f64; 2]>::with_capacity(data.nrows() / 10); // 1/10 is bg
    for i in all_pairs.into_iter() {
        for j in i.0 {
            fg.push(j);
        }
        for j in i.1 {
            bg.push(j)
        }
    }

    (fg, bg)
}
