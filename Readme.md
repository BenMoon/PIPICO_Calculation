Python module to calculate a PIPICO / Covariance map used in VMI data analysis.
This is work in progress and by no means stable, yet.

# Install
- install `rust` on your machine (https://www.rust-lang.org/tools/install)
- install `maturin` in your Python environment (e. g. `conda install maturin`)
- compile code with something like `maturin develop --release`

# Usage
The data is assumed to be a 2D array with your shots / triggers / data to correlate with one
another, to be in a row.

The return of the both functions are coincidence maps and thus uncorrelated events need to be
subtracted separately.
Returns are counts and isn't normalized to the number of triggers.

You can provide a square numpy array (values in rows need to be sorted), or a list with of lists
(values don't need to be sorted) which I usually generate from my data
like
```python
import pipico

bins = 5000
hist_min = 0
hist_max = 5

# calculate correlated events
a = list(df.groupby(['nr'])['tof'].apply(list))
pipico_map = pipico.pipico_lists(a, bins, hist_min, hist_max)

# calculate un-correlated events
h_1d = np.histogram(list(df['tof']), bins=bins, range=(hist_min, hist_max))[0] / len(a)
pipico_bg = h_1d[:, None] * h_1d[None, :]
j1d = np.arange(bins)
jx, jy = np.meshgrid(j1d, j1d, indexing="ij")
pipico_bg[jx <= jy] = 0
pipico_bg[jx <= jy] = 0

# subtract correlated from uncorrelated map:
pipico_cov = pipico_map / len(a) - pipico_bg
```

[Bins created](https://docs.rs/ndhistogram/0.6.2/ndhistogram/axis/struct.Uniform.html):
> An axis with N equally spaced, equal sized, bins between (low, high]. Below (above) this range is an underflow (overflow) bin. Hence this axis has N+2 bins.

unlike [numpy](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html?highlight=histogram#numpy.histogram), where:
> All but the last (righthand-most) bin is half-open. In other words, if bins is:
> 
> `[1, 2, 3, 4]`
>
> then the first bin is `[1, 2)` (including 1, but excluding 2) and the second `[2, 3)`. The last bin, however, is `[3, 4]`, which includes 4.