//#![feature(iter_collect_into)]

use bit_set::BitSet;
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

/// calculate a covariance map
#[pymodule]
fn pipico(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// DEPRECATED, toy code remaining for demonstration purposes
    /// calculate a covariance map
    /// x: list of lists with a ToF trace in every row, pre-sorting not required
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    /// ```python
    /// import pipico
    ///
    /// bins = 5000
    /// hist_min = 0
    /// hist_max = 5
    ///
    /// # calculate correlated events
    /// a = list(df.groupby(['nr'])['tof'].apply(list))
    /// pipico_map = pipico.pipico_lists(a, bins, hist_min, hist_max)
    ///
    /// # calculate un-correlated events
    /// h_1d = np.histogram(list(df['tof']), bins=bins, range=(hist_min, hist_max))[0] / len(a)
    /// pipico_bg = h_1d[:, None] * h_1d[None, :]
    /// j1d = np.arange(bins)
    /// jx, jy = np.meshgrid(j1d, j1d, indexing="ij")
    /// pipico_bg[jx <= jy] = 0
    /// pipico_bg[jx <= jy] = 0
    ///
    /// # subtract correlated from uncorrelated map:
    /// pipico_cov = pipico_map / len(a) - pipico_bg
    /// ```
    #[pyfn(m)]
    fn pipico_lists(
        py: Python<'_>,
        mut x: Vec<Vec<f64>>,
        n_bins: usize,
        min: f64,
        max: f64,
    ) -> PyResult<&PyArray2<f64>> {
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
        let a_hist: Array2<f64> = Array1::from_iter(hist.values().copied())
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

        let a_hist: Array2<f64> = Array1::from_iter(hist.values().copied())
            .into_shape((n_bins + 2, n_bins + 2))
            .unwrap();

        Ok(a_hist
            .slice(s![1..n_bins + 1, 1..n_bins + 1])
            .to_pyarray(py))
    }

    /// Extract ion pairs fulfilling the condition of a 3D momentum conservation using a relative
    /// threshold for the momentum conservation.
    ///
    /// # Arguments
    ///
    /// * `py` - A Python context required for creating PyArray2 objects.
    /// * `x` - Input data as a 2D array of 64-bit floating-point numbers.
    /// * `momentum_cut` - A floating-point number used to determine the criteria for filtering
    ///    pairs representing a Newton-sphere.
    ///
    /// # Returns
    ///
    /// A tuple containing two PyArray2 objects representing covariance pairs and background pairs.
    ///
    /// # Input Data Format:
    ///
    /// The input data (`x`) is expected to be a 2D array, where each row represents a set of
    /// measurements.
    /// The first column is assumed to contain trigger numbers, and the subsequent columns contain
    /// measurement data.
    /// 2D array containing the data [trigger nr, index, px, py, pz, tof] index is the pandas
    /// DataFrame index.
    ///
    /// # Parallel Processing
    ///
    /// This function divides the input data into chunks and processes them in parallel
    /// optimizing performance on multi-core CPUs.
    ///
    /// # Performance Considerations
    ///
    /// This function is designed for performance optimization, utilizing parallel processing and
    /// efficient data handling to compute covariance pairs and background pairs from large datasets.
    ///
    /// # Example Usage
    ///
    /// Here is an example of how to use this function in a Python application:
    ///
    /// ```python
    /// import pipico
    /// # import numpy, pandas, polars
    ///
    /// def sort_pairs(df: pd.DataFrame, pairs: np.array, pairs_bg: np.array) -> (pd.DataFrame, pd.DataFrame):
    ///     # forground
    ///     df_p1 = df.loc[pairs[:, 0]].copy()
    ///     df_p2 = df.loc[pairs[:, 1]].copy()
    ///     df_p1.reset_index(drop=True, inplace=True)
    ///     df_p2.reset_index(drop=True, inplace=True)
    ///     df_p1 = df_p1.add_suffix("1")
    ///     df_p2 = df_p2.add_suffix("2")
    ///     df_pairs = pd.concat([df_p1, df_p2], axis=1)
    ///
    ///     # Background
    ///     df_p1 = df.loc[pairs_bg[:, 0]].copy()
    ///     df_p2 = df.loc[pairs_bg[:, 1]].copy()
    ///     df_p1.reset_index(drop=True, inplace=True)
    ///     df_p2.reset_index(drop=True, inplace=True)
    ///     df_p1 = df_p1.add_suffix("1")
    ///     df_p2 = df_p2.add_suffix("2")
    ///     df_pairs_bg = pd.concat([df_p1, df_p2], axis=1)
    ///
    ///     return df_pairs, df_pairs_bg
    ///
    /// # prepare the data
    /// df_snowman.sort_values(['trigger nr', 'tof'], inplace=True)
    /// df_snowman.reset_index(inplace=True, drop=True)
    /// df_snowman['idx'] = df_snowman.index
    ///
    /// # Call the Rust function
    /// da = pl.from_pandas(df)[['trigger nr', 'idx', 'p_x', 'p_y', 'p_z', 'tof', 'mz']].to_numpy()
    /// pairs_fg, pairs_bg = pipico.get_covar_pairs(x=da, momentum_cut=3)
    ///
    /// # Use the resulting covariance and background pairs as needed
    /// df_pairs_fg, df_pairs_bg = sort_pairs(df_snowman, pairs_fg, pairs_bg)
    /// bins = np.linspace(15, 21, 200)
    /// xy_hist, x_bins, y_bins = np.histogram2d(df_pairs_fg['mz1'], df_pairs_fg['mz2'], bins=bins)
    /// xy_hist_bg, x_bins, y_bins = np.histogram2d(df_pairs_bg['mz1'], df_pairs_bg['mz2'], bins=bins)
    /// rasterize(hv.Image((xy_hist - xy_hist_bg).T[::-1], bounds=[x_bins[0], y_bins[0], x_bins[-1], y_bins[-1]]).opts(xlabel='m_1/q_1', ylabel='m_2/q_2', title=f'relative cut={p_cut}\n{git_info()}'))
    /// ```
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

    /// Extract ion pairs fulfilling a 3D momentum conservation condition.
    ///
    /// This function processes a 2D array of data, where each row contains information
    /// about trigger numbers, indices, momenta (px, py, pz), time-of-flight (tof), and mass.
    ///
    /// The `mass_momentum_cut` parameter is used to specify momentum cuts based on mass ranges.
    /// It should be a 2D array with each row having five values: [min_mass, max_mass, min_px, max_px, max_momentum^2].
    ///
    /// The `default_momentum_cut` parameter defines a momentum sphere for all other masses.
    ///
    /// The function extracts ion pairs that satisfy the specified momentum conservation conditions,
    /// and it performs these operations in parallel to leverage multi-core processing.
    ///
    /// # Arguments
    ///
    /// - `x`: A 2D array containing the input data: [trigger nr, index, px, py, pz, tof, mass]
    /// - `mass_momentum_cut`: A 2D array specifying momentum cuts based on mass ranges.
    ///   `px+py+pz <= mass_momentum_cut` for masses define in this array, e.g.
    /// - `default_momentum_cut`: A floating-point value for the default momentum cut.
    ///    momentum sphere which applies for all other masses
    ///
    /// # Returns
    ///
    /// A tuple containing two 2D arrays: the first array contains foreground ion pairs,
    /// and the second array contains background ion pairs.
    ///
    /// Both arrays are represented as arrays of floating-point values.
    ///
    /// # Example
    ///
    /// ```python
    /// import pipico
    /// # import numpy, pandas, polars
    ///
    /// def sort_pairs(df: pd.DataFrame, pairs: np.array, pairs_bg: np.array) -> (pd.DataFrame, pd.DataFrame):
    ///     # forground
    ///     df_p1 = df.loc[pairs[:, 0]].copy()
    ///     df_p2 = df.loc[pairs[:, 1]].copy()
    ///     df_p1.reset_index(drop=True, inplace=True)
    ///     df_p2.reset_index(drop=True, inplace=True)
    ///     df_p1 = df_p1.add_suffix("1")
    ///     df_p2 = df_p2.add_suffix("2")
    ///     df_pairs = pd.concat([df_p1, df_p2], axis=1)
    ///
    ///     # Background
    ///     df_p1 = df.loc[pairs_bg[:, 0]].copy()
    ///     df_p2 = df.loc[pairs_bg[:, 1]].copy()
    ///     df_p1.reset_index(drop=True, inplace=True)
    ///     df_p2.reset_index(drop=True, inplace=True)
    ///     df_p1 = df_p1.add_suffix("1")
    ///     df_p2 = df_p2.add_suffix("2")
    ///     df_pairs_bg = pd.concat([df_p1, df_p2], axis=1)
    ///
    ///     return df_pairs, df_pairs_bg
    ///
    /// # prepare the data
    /// df_snowman.sort_values(['trigger nr', 'tof'], inplace=True)
    /// df_snowman.reset_index(inplace=True, drop=True)
    /// df_snowman['idx'] = df_snowman.index
    ///
    /// # Call the Rust function
    /// da = pl.from_pandas(df)[['trigger nr', 'idx', 'p_x', 'p_y', 'p_z', 'tof', 'mz']].to_numpy()
    /// pairs_fg, pairs_bg = pipico.get_covar_pairs_fixed_cut(x=da, momentum_cut=3e4)
    ///
    /// # Use the resulting covariance and background pairs as needed
    /// df_pairs_fg, df_pairs_bg = sort_pairs(df_snowman, pairs_fg, pairs_bg)
    /// bins = np.linspace(15, 21, 200)
    /// xy_hist, x_bins, y_bins = np.histogram2d(df_pairs_fg['mz1'], df_pairs_fg['mz2'], bins=bins)
    /// xy_hist_bg, x_bins, y_bins = np.histogram2d(df_pairs_bg['mz1'], df_pairs_bg['mz2'], bins=bins)
    /// rasterize(hv.Image((xy_hist - xy_hist_bg).T[::-1], bounds=[x_bins[0], y_bins[0], x_bins[-1], y_bins[-1]]).opts(xlabel='m_1/q_1', ylabel='m_2/q_2', title=f'relative cut={p_cut}\n{git_info()}'))
    /// ```
    ///
    /// The `fg_pairs` array will contain foreground ion pairs, and the `bg_pairs` array will
    /// contain background ion pairs.
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
                                .filter(|&x| {
                                    let p_sum = (*x[2] + *px).powf(2.)
                                        + (*x[3] + *py).powf(2.)
                                        + (*x[4] + *pz).powf(2.);
                                    let m2 = *x[6];

                                    // apply for mass specific momentum cut
                                    for i in 0..mp_cut.nrows() {
                                        if m1 >= &mp_cut[[i, 0]]
                                            && m1 < &mp_cut[[i, 1]]
                                            && m2 >= &mp_cut[[i, 2]]
                                            && m2 < &mp_cut[[i, 3]]
                                        {
                                            return p_sum <= mp_cut[[i, 4]];
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
                                .filter(|&x| {
                                    let p_sum = (*x[2] + px).powf(2.)
                                        + (*x[3] + py).powf(2.)
                                        + (*x[4] + pz).powf(2.);
                                    let m2 = *x[6];

                                    // apply for mass specific momentum cut
                                    for i in 0..mp_cut.nrows() {
                                        if m1 >= mp_cut[[i, 0]]
                                            && m1 < mp_cut[[i, 1]]
                                            && m2 >= &mp_cut[[i, 2]]
                                            && m2 < &mp_cut[[i, 3]]
                                        {
                                            return p_sum <= mp_cut[[i, 4]];
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

    /// Generates a unique set of background indices using a random number generator and a given maximum index.
    ///
    /// This function generates a set of background indices that are not present in the provided `idx_trg_frame`.
    /// It uses a random number generator `rng` to generate unique background indices within the range [0, `max_index`).
    ///
    /// # Arguments
    ///
    /// - `rng`: A mutable reference to a random number generator (e.g., `ThreadRng`).
    /// - `idx_trg_frame`: A view of the trigger frame indices.
    /// - `max_index`: The maximum index value (exclusive) for generating background indices.
    ///
    /// # Returns
    ///
    /// A `Vec<usize>` containing the generated unique background indices.
    ///
    /// # Example
    ///
    /// ```rust
    /// use rand::thread_rng;
    /// use ndarray::{arr1, ArrayView1};
    /// use pipico::get_bg_idx;
    ///
    /// let mut rng = thread_rng();
    /// let trigger_frame_index = arr1(&[&&1.0, &&3.0, &&5.0, &&7.0, &&9.0, &&11.0]);
    /// let max_index: usize = 20;
    /// let result = get_bg_idx(&mut rng, trigger_frame_index.view(), max_index);
    ///
    /// // The `result` vector now contains unique background indices.
    /// ```
    ///
    /// Note that the `get_bg_idx` function guarantees that the returned background
    /// indices are unique and not present in `idx_trg_frame`.
    ///
    /// # Performance
    ///
    /// This function is optimized for performance and efficiency, making it suitable for generating
    /// background indices in applications requiring low latency and memory usage.
    ///
    /// # Panics
    ///
    /// This function may panic if `max_index` is less than the number of elements in `idx_trg_frame`.
    ///
    /// # Safety
    ///
    /// This function assumes that the input parameters are valid and that the random number
    /// generator is correctly initialized.
    fn get_bg_idx(
        rng: &mut ThreadRng,
        idx_trg_frame: ArrayBase<ViewRepr<&&&f64>, Dim<[usize; 1]>>,
        max_index: usize,
    ) -> Vec<usize> {
        let mut idx_bg = BitSet::new();
        let mut result = Vec::with_capacity(idx_trg_frame.len());

        for _ in 0..idx_trg_frame.len() {
            let mut bg: usize;
            loop {
                bg = rng.gen_range(0..max_index);
                if !idx_bg.contains(bg) {
                    idx_bg.insert(bg);
                    result.push(bg);
                    break;
                }
            }
        }

        result
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

    for _ in 0..idx_trg_frame.len() {
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
    let mut idx_set = HashSet::new();
    let mut idx_bg = HashSet::new();
    let mut result = Vec::with_capacity(idx_trg_frame.len());

    for &&&val in idx_trg_frame.iter() {
        idx_set.insert(val as usize);
    }

    while result.len() < idx_trg_frame.len() {
        let bg = rng.gen_range(0..max_index);
        if !idx_set.contains(&bg) && idx_bg.insert(bg) {
            result.push(bg);
        }
    }

    result
}

/// optimised rust implementation for benchmarking

pub fn get_bg_idx_set_optimized(
    rng: &mut ThreadRng,
    idx_trg_frame: ArrayBase<ViewRepr<&&&f64>, Dim<[usize; 1]>>,
    max_index: usize,
) -> Vec<usize> {
    let idx_set: BitSet = idx_trg_frame
        .iter()
        .map(|&&&x| x as usize)
        .collect::<BitSet>();

    let mut idx_bg = BitSet::with_capacity(idx_trg_frame.len());
    let mut result = Vec::with_capacity(idx_trg_frame.len());

    while result.len() < idx_trg_frame.len() {
        let bg: usize = rng.gen_range(0..max_index);
        if !idx_set.contains(bg) && !idx_bg.contains(bg) {
            idx_bg.insert(bg);
            result.push(bg);
        }
    }

    result
}

/// rust implementation for benchmarking
// data needs to be sorted along triggers
// this should already work
// works with 2D array
pub fn ndarray_filter_momentum_bench_2D(
    data: Array2<f64>,
    //) -> PyResult<PyDataFrame> {
) -> Array2<f64> {
    //let filter_delta = 0.01;
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

    let a_hist: Array2<f64> = Array1::from_iter(cov_hist.values().copied())
        .into_shape((n_bins + 2, n_bins + 2))
        .unwrap();

    a_hist.slice(s![1..n_bins + 1, 1..n_bins + 1]).to_owned()
}

/// rust implementation for benchmarking
pub fn get_pairs_bench(data: Array2<f64>) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    // let filter_delta = 0.01;
    // let n_bins = 10;
    // let hist_min = 0.;
    // let hist_max = 10.;

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
                            .filter(|&x| {
                                let p_sum = (*x[2] + *px).powf(2.)
                                    + (*x[3] + *py).powf(2.)
                                    + (*x[4] + *pz).powf(2.);
                                let m2 = *x[6];

                                // apply for mass specific momentum cut
                                for i in 0..mp_cut.nrows() {
                                    if m1 >= &mp_cut[[i, 0]]
                                        && m1 < &mp_cut[[i, 1]]
                                        && m2 >= &mp_cut[[i, 2]]
                                        && m2 < &mp_cut[[i, 3]]
                                    {
                                        return p_sum <= mp_cut[[i, 4]];
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
                            .filter(|&x| {
                                let p_sum = (*x[2] + px).powf(2.)
                                    + (*x[3] + py).powf(2.)
                                    + (*x[4] + pz).powf(2.);
                                let m2 = *x[6];

                                // apply for mass specific momentum cut
                                for i in 0..mp_cut.nrows() {
                                    if m1 >= mp_cut[[i, 0]]
                                        && m1 < mp_cut[[i, 1]]
                                        && m2 >= &mp_cut[[i, 2]]
                                        && m2 < &mp_cut[[i, 3]]
                                    {
                                        return p_sum <= mp_cut[[i, 4]];
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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_get_bg_idx_set() {
        // Create a mock trigger frame index
        let trigger_frame_index = arr1(&[&&1.0, &&3.0, &&5.0, &&7.0, &&9.0, &&11.0]);

        let max_index: usize = 20;

        // Initialize the RNG
        let mut rng = thread_rng();

        // Call the function under test
        let result = get_bg_idx_set(&mut rng, trigger_frame_index.view(), max_index);

        // Convert the trigger_frame_index to a HashSet for efficient containment check
        let trigger_frame_set: std::collections::HashSet<u64> =
            trigger_frame_index.iter().map(|&&&x| x as u64).collect();

        // Assert that the result contains the correct number of unique indices
        assert_eq!(result.len(), trigger_frame_index.len());

        // Assert that the result contains indices that are not in the trigger frame index
        for bg in &result {
            assert!(!trigger_frame_set.contains(&(*bg as u64)));
        }

        // Assert that the result only contains unique indices
        assert_eq!(
            result.len(),
            result
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
    }

    #[test]
    fn test_get_bg_idx_set_opt() {
        // Create a mock trigger frame index
        let trigger_frame_index = arr1(&[&&1.0, &&3.0, &&5.0, &&7.0, &&9.0, &&11.0]);

        let max_index: usize = 20;

        // Initialize the RNG
        let mut rng = thread_rng();

        // Call the function under test
        let result = get_bg_idx_set_optimized(&mut rng, trigger_frame_index.view(), max_index);

        // Convert the trigger_frame_index to a HashSet for efficient containment check
        let trigger_frame_set: std::collections::HashSet<u64> =
            trigger_frame_index.iter().map(|&&&x| x as u64).collect();

        // Assert that the result contains the correct number of unique indices
        assert_eq!(result.len(), trigger_frame_index.len());

        // Assert that the result contains indices that are not in the trigger frame index
        dbg!(&result);
        for bg in &result {
            assert!(!trigger_frame_set.contains(&(*bg as u64)));
        }

        // Assert that the result only contains unique indices
        assert_eq!(
            result.len(),
            result
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .len()
        );
    }
}
