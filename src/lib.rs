use std::usize;

use ndarray::{array, s, Array, Array1, Array2, ArrayBase, ArrayView2, Dim, Slice, ViewRepr};
use ndhistogram::{axis::Axis, axis::Category, axis::Uniform, ndhistogram, value::Mean, Histogram};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::{exceptions::PyRuntimeError, pymodule, types::PyModule, PyErr, PyResult, Python};

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
        nbins: usize,
        min: f64,
        max: f64
    ) -> PyResult<&'py PyArray2<f64>> {
        let data = x.as_array();
        let Npart = data.shape()[1];
        let Nshots = data.shape()[0];

        // sort data into histogram iterating through data 2D array
        // initialize empty 2D histogram
        let mut hist = ndhistogram!(
            Uniform::<f64>::new(nbins, min, max),
            Uniform::<f64>::new(nbins, min, max)
        );
        let (mut p1, mut p2) = (0, 0);
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
            .into_shape((nbins+2, nbins+2))
            .unwrap();

        Ok(a_hist.slice(s![1..nbins+1, 1..nbins+1]).to_pyarray(py))
    }

    Ok(())
}
