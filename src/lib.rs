use std::usize;

use ndarray::{s, Array1, Array2};
use ndhistogram::{axis::Uniform, ndhistogram, Histogram};
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

/// calculate a covariance map
#[pymodule]
fn pipico(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    /// calculate a covariance map
    /// x: 2D array with a ToF trace in every row
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    #[pyfn(m)]
    fn calc_pipico<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        n_bins: usize,
        min: f64,
        max: f64
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
            .into_shape((n_bins+2, n_bins+2))
            .unwrap();

        Ok(a_hist.slice(s![1..n_bins+1, 1..n_bins+1]).to_pyarray(py))
    }

    /// calculate a covariance map
    /// x: list of lists with a ToF trace in every row
    /// nbins: number of bins for map
    /// min: histogram min
    /// max: histogram max
    #[pyfn(m)]
    fn pipico_lists<'py>(
        py: Python<'py>,
        x: Vec<Vec<f64>>,
        n_bins: usize,
        min: f64,
        max: f64
    ) -> PyResult<&'py PyArray2<f64>> {
        //let data = x.as_array();
        //let Nshots = data.shape()[0];

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
         */
        let a_hist: Array2<f64> = Array1::from_iter(hist.values().map(|v| *v).into_iter())
            .into_shape((n_bins+2, n_bins+2))
            .unwrap();

        Ok(a_hist.slice(s![1..n_bins+1, 1..n_bins+1]).to_pyarray(py))
         
        //let dummy = Array::<f64,_>::zeros((10, 10));
        //Ok(dummy.into_pyarray(py))
    }
    Ok(())
}
