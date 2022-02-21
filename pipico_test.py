import numpy as np
import time
import pipico
import matplotlib.pyplot as plt

nbins = 10
Nshots = 100
Npart = 10

# simulate some data
data = np.zeros((Nshots, Npart))
for i in range(Nshots):
    dt1 = np.random.uniform(-0.1, 0.1)
    data[i][0] = 3 - dt1
    data[i][1] = 6 + dt1
    data[i][2:] = np.random.uniform(0, 10, Npart - 2)
    data[i] = np.sort(data[i])

# get histogram from module
tstart = time.time()
hist = pipico.calc_pipico(data, nbins, 0, 10)
tstop = time.time()
print(f"time it took {tstop - tstart:.5f} s")

tstart = time.time()
h_2d = np.zeros((nbins, nbins))
for row in data:
    h_1d_i = np.histogram(row, bins=nbins, range=(0, 10))[0]
    h_2d += h_1d_i[:, None] * h_1d_i[None, :]
tstop = time.time()
print(f"time it took {tstop - tstart:.5f} s")

