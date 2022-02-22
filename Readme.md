Python module to calculate a PIPICO / Covariance map.

# Install
- install `rust` on your machine (https://www.rust-lang.org/tools/install)
- install `maturin` in your Python environment (e. g. `conda install maturin`)
- compile code with something like `maturin develop -release`

# Usage
The data is assumed to be a 2D array with your shots / triggers / data to correlate with one
another, to be in a row.

You can provide a square numpy array, or a list with of lists which I usually generate from my data
like
```python
groups = list(data.groupby('trigger number')['tof'].apply(list))
hist = pipico.pipico_list(groups)
```