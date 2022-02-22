import numpy as np
import time
import pipico
import matplotlib.pyplot as plt

nbins = 1000
Nshots = 300_000
Npart = 100

# simulate some data
data = np.zeros((Nshots, Npart))
for i in range(Nshots):
    dt1 = np.random.uniform(-0.1, 0.1)
    data[i][0] = 3 - dt1
    data[i][1] = 6 + dt1
    data[i][2:] = np.random.uniform(0, 10, Npart - 2)
    #data[i] = np.sort(data[i])
'''
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

# test list of lists approach
# a = list(data_cent.groupby(['nr'])['tof'].apply(list))
# [[float(f'{j:.1f}') for j in i[:10]] for i in a[:10]]
data = [[2.5, 3.2, 3.2, 3.2, 3.2, 3.5, 3.4, 3.6, 4.1, 4.9],
        [1.5, 3.2, 3.2, 3.3, 3.2, 3.4, 4.1, 3.2],
        [1.5, 1.5, 1.5, 1.5, 1.5, 3.2, 3.6, 3.2, 3.2],
        [1.5, 1.5, 1.5, 1.5, 2.5, 3.4, 4.1],
        [2.5, 3.2, 3.2, 3.5, 3.2, 3.6, 4.1, 5.5, 5.5, 5.5],
        [1.5, 1.5, 3.2, 3.2, 3.2, 4.1, 3.5, 3.5, 3.2],
        [1.5, 3.2, 3.4, 3.6, 4.1, 4.1, 4.1, 4.1],
        [1.5, 1.5, 1.6, 1.5, 3.2, 3.4, 3.2, 3.1, 3.2],
        [1.5, 3.2, 3.2, 3.2, 4.2, 4.1, 3.2, 3.4, 3.5, 5.5],
        [1.5, 1.5, 1.5, 1.5, 1.5, 2.6, 3.2, 3.2, 3.2]]
data.append(data[0][::-1])
data.append(data[0][::-1])
data.append(data[0][::-1])
'''

tstart = time.time()
hist = pipico.pipico_lists(data, 10, 0, 8)
tstop = time.time()
print(f"time it took {tstop - tstart:.5f} s")

'''
tstart = time.time()
h_2d = np.zeros((10, 10))
for row in data:
    h_1d_i = np.histogram(row, bins=10, range=(0, 8))[0]
    h_2d += h_1d_i[:, None] * h_1d_i[None, :]
tstop = time.time()
print(f"time it took {tstop - tstart:.5f} s")
'''