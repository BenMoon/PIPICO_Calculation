use std::usize;

use itertools::DedupBy;
use ndarray::{s, Array1, Array2, Data};
use ndarray::prelude::*;
use ndhistogram::{axis::Uniform, ndhistogram, Histogram};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use polars::lazy::dsl::apply_multiple;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use polars::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::PyDataFrame;

use rayon::prelude::*;

/// calculate a covariance map
#[pymodule]
fn pipico(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// calculate a covariance map
    /// x: 2D array with a ToF trace in every row, requires the rows to sorted
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
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
    fn polars_filter_momentum<'py>(
        py: Python<'py>,
        pydf: PyDataFrame,
        col_grp: &str,
        col_pipico: &str,
        col_mask: &str,
        n_bins: usize,
        min_tof: f64,
        max_tof: f64,
    //) -> PyResult<PyDataFrame> {
    ) -> PyResult<&'py PyArray2<f64>> {
        let df: DataFrame = pydf.into();

        /*
        let grouped = df.clone().lazy()
            .groupby(["trigger nr"])
            .agg([col(col_pipico).sort(false), col(col_mask)])
            .collect().expect("something went wrong with grouping");
        */

        let compute_covariance = move |s: &mut [Series]| {
            let mut hist = ndhistogram!(
                Uniform::<f64>::new(n_bins, min_tof, max_tof),
                Uniform::<f64>::new(n_bins, min_tof, max_tof)
            );
    
            let df = df!("col_cov"  => s[0].clone(),
                                    "col_mask" => s[1].clone()).unwrap();
            
            // filter data accordingly
            /*
            let mask = df
                .lazy()
                .filter(col("col_mask").gt(lit(5000.0)).and(col("col_mask").lt(20_000.0)))
                .select([col("col_cov")])
                .collect()
                .expect("Could filter col_mask");
             */
        
            // https://docs.rs/polars/0.26.1/polars/docs/eager/index.html#extracting-data
            let ca = df.column("col_cov").unwrap().f64().unwrap();
            //dbg!(&ca);
            let mut p2 = 0;
            for (p1, x) in ca.into_iter().enumerate() {
                p2 = p1 + 1;
                match x {
                    None => panic!("Not a float"),
                    Some(x) =>
                        // filter needs to go here...
                        while p2 < ca.len() {
                            hist.fill(&(x, ca.get(p2).unwrap()));
                            //println!("{:?}={:?}, {:?}={:?}", p1, x, p2, ca.get(p2).unwrap());
                            p2 += 1;
                    }
                }
            }
            let a = hist.values().map(|v| *v).collect::<Vec<f64>>();
            Ok(Series::from_vec("hist", a))
        };

        // group df according to 'trigger nr'
        let grouped = df
            .clone()
            .lazy()
            .groupby([col_grp])
            .agg([apply_multiple(
                compute_covariance,
                [col(col_pipico).sort(false), col(col_mask)],
                GetOutput::from_type(DataType::Float64),
                false,
            ).alias("histogram")])
            .select([col("histogram")])
            .collect()
            .expect("msg");
        let number_groups = grouped.height();
        
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

    fn compute_covariance_fn(s: &mut [Series]) -> PolarsResult<Series> {
        // sort data into histogram iterating through data 2D array
        // initialize empty 2D histogram
        let n_bins = 1000;
        let min = 15.0;
        let max = 20.0;
        let mut hist = ndhistogram!(
            Uniform::<f64>::new(n_bins, min, max),
            Uniform::<f64>::new(n_bins, min, max)
        );

        let df = df!("col_cov"  => s[0].clone(),
                                "col_mask" => s[1].clone()).unwrap();
        
        /*
        let mask = df
            .lazy()
            .filter(col("col_mask").gt(lit(5000.0)).and(col("col_mask").lt(20_000.0)))
            .select([col("col_cov")])
            .collect()
            .expect("Could filter col_mask");
         */
    
        // https://docs.rs/polars/0.26.1/polars/docs/eager/index.html#extracting-data
        let ca = df.column("col_cov").unwrap().f64().unwrap();
        let mut p2 = 0;
        for (p1, x) in ca.into_iter().enumerate() {
            p2 = p1 + 1;
            match x {
                None => panic!("Not a float"),
                Some(x) => 
                    // filter actually needs to go here...
                    while p2 < ca.len() {
                        println!("{:?}, {:?}", p1, p2);
                        p2 += 1;
                }
            }
        }

        let s1 = Series::new("float", &[Some(1.0), Some(f32::NAN), Some(3.0)]);
        Ok(s1)
    }

    Ok(())
}
