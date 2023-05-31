import time
import numpy as np
import pandas as pd
import polars as pl

import pipico


def test_pipico_list():
    data = [
        [2.5, 3.2, 3.2, 3.2, 3.2, 3.5, 3.4, 3.6, 4.1, 4.9],
        [1.5, 3.2, 3.2, 3.3, 3.2, 3.4, 4.1, 3.2],
        [1.5, 1.5, 1.5, 1.5, 1.5, 3.2, 3.6, 3.2, 3.2],
        [1.5, 1.5, 1.5, 1.5, 2.5, 3.4, 4.1],
        [2.5, 3.2, 3.2, 3.5, 3.2, 3.6, 4.1, 5.5, 5.5, 5.5],
        [1.5, 1.5, 3.2, 3.2, 3.2, 4.1, 3.5, 3.5, 3.2],
        [1.5, 3.2, 3.4, 3.6, 4.1, 4.1, 4.1, 4.1],
        [1.5, 1.5, 1.6, 1.5, 3.2, 3.4, 3.2, 3.1, 3.2],
        [1.5, 3.2, 3.2, 3.2, 4.2, 4.1, 3.2, 3.4, 3.5, 5.5],
        [1.5, 1.5, 1.5, 1.5, 1.5, 2.6, 3.2, 3.2, 3.2],
    ]
    data.append(data[0][::-1])
    data.append(data[0][::-1])
    data.append(data[0][::-1])

    hist = pipico.pipico_lists(data, 10, 1, 6)
    should_be = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 75.0, 0.0, 27.0, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 11.0, 0.0, 10.0, 64.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 13.0, 0.0, 6.0, 52.0, 18.0, 7.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 20.0, 8.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 3.0, 14.0, 7.0, 5.0, 0.0, 0.0, 3.0],
        ]
    )
    assert (hist == should_be).all()


def test_pipico_polars_filter_momentum():
    df = pl.from_pandas(pd.read_feather("pipico_test_data.feather"))
    data = [
        [2.5, 3.2, 3.2, 3.2, 3.2, 3.5, 3.4, 3.6, 4.1, 4.9],
        [1.5, 3.2, 3.2, 3.3, 3.2, 3.4, 4.1, 3.2],
        [1.5, 1.5, 1.5, 1.5, 1.5, 3.2, 3.6, 3.2, 3.2],
        [1.5, 1.5, 1.5, 1.5, 2.5, 3.4, 4.1],
        [2.5, 3.2, 3.2, 3.5, 3.2, 3.6, 4.1, 5.5, 5.5, 5.5],
        [1.5, 1.5, 3.2, 3.2, 3.2, 4.1, 3.5, 3.5, 3.2],
        [1.5, 3.2, 3.4, 3.6, 4.1, 4.1, 4.1, 4.1],
        [1.5, 1.5, 1.6, 1.5, 3.2, 3.4, 3.2, 3.1, 3.2],
        [1.5, 3.2, 3.2, 3.2, 4.2, 4.1, 3.2, 3.4, 3.5, 5.5],
        [1.5, 1.5, 1.5, 1.5, 1.5, 2.6, 3.2, 3.2, 3.2],
    ]
    data.append(data[0][::-1])
    data.append(data[0][::-1])
    data.append(data[0][::-1])

    trigger_nrs = [num for vec in [len(row) * [i] for i, row in enumerate(data)] for num in vec]
    x = [i for vec in data for i in vec]

    df = pl.from_pandas(
        pd.DataFrame(np.column_stack((trigger_nrs, x, x)), columns=["trigger nr", "mz", "p_abs"])
    )

    hist = pipico.polars_filter_momentum(
        pydf=df[["trigger nr", "mz", "p_abs"]],
        col_grp="trigger nr",
        col_pipico="mz",
        col_mask="p_abs",
        filter_delta=10,  # has no effect as it is
        n_bins=10,
        hist_min=1,
        hist_max=6,
    )

    should_be = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 75.0, 0.0, 27.0, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 11.0, 0.0, 10.0, 64.0, 6.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 13.0, 0.0, 6.0, 52.0, 18.0, 7.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 20.0, 8.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 3.0, 14.0, 7.0, 5.0, 0.0, 0.0, 3.0],
        ]
    )
    assert (hist == should_be).all(), hist


def test_pipico_polars_filter_relative_momentum():
    # df = pl.from_pandas(
    #    pd.DataFrame(pd.read_feather('test_data.feather'), columns=['trigger nr', 'tof', 'px', 'py']))
    # simulate data
    n_bins = 10
    n_shots = 1_000
    n_parts = 100

    tof_min = 4
    tof_max = 4.45
    df = gen_data(n_shots, n_parts)
    #df = pl.read_parquet("test_data.parquet")
    da = pl.from_pandas(df)[['trigger nr', 'idx', 'px', 'py', 'pz', 'tof']].to_numpy()

    # get pairs from rust
    start = time.time()
    pairs_fg, pairs_bg = pipico.get_covar_pairs(x=da, momentum_cut=0.0025)
    stop = time.time()
    time_rs = stop - start
    print(f"Rust numpy took: {(time_rs)*1e3} ms")

    # get pairs from python implementation
    start = time.time()
    hist2d, hist2d_bg, pairs_py_fg, pairs_py_bg = filter_covariance_py(df, n_bins=n_bins, tof_min=tof_min, tof_max=tof_max)
    stop = time.time()
    time_py = stop - start
    pairs_py_fg = np.array(pairs_py_fg)
    pairs_py_bg = np.array(pairs_py_bg)
    print(f"Python took: {(time_py)} s")
    print(f'Rust {(time_py / time_rs)}x faster')
    
    assert pairs_fg.all() == pairs_py_fg.all()
    df_pairs_fg, df_pairs_bg = sort_pairs(df, pairs_fg, pairs_bg)
    assert ((df_pairs_fg['tof2'] - df_pairs_fg['tof1']) >= 0).all(), "TOF1 <= TOF2 needs to be given"
    assert ((df_pairs_bg['tof2'] - df_pairs_bg['tof1']) >= 0).all(), "TOF1 <= TOF2 needs to be given"
    assert (df_pairs_fg['trigger nr1'] == df_pairs_fg['trigger nr2']).all(), "triggers for pairs are not same"
    #assert (df_pairs_bg['trigger nr1'] != df_pairs_bg['trigger nr2']).all(), "triggers for pairs in bg are not same"
    df_pairs_py_fg, df_pairs_py_bg = sort_pairs(df, pairs_py_fg, pairs_py_bg)
    assert ((df_pairs_py_fg['tof2'] - df_pairs_py_fg['tof1']) >= 0).all(), "TOF1 <= TOF2 needs to be given"
    assert ((df_pairs_py_bg['tof2'] - df_pairs_py_bg['tof1']) >= 0).all(), "TOF1 <= TOF2 needs to be given"
    assert df_pairs_fg.all().all() == df_pairs_py_fg.all().all()

    bins = np.linspace(df.tof.min(), df.tof.max(), 10)
    xy_hist, _, _ = np.histogram2d(df_pairs_fg["tof1"], df_pairs_fg["tof2"], bins=bins)
    xy_hist_bg, _, _ = np.histogram2d(df_pairs_bg["tof1"], df_pairs_bg["tof2"], bins=bins)
    print('rust fg')
    print(np.int_(xy_hist))
    print('rust bg')
    print(np.int_(xy_hist_bg))

    xy_hist_py, _, _ = np.histogram2d(df_pairs_py_fg["tof1"], df_pairs_py_fg["tof2"], bins=bins)
    xy_hist_bg_py, _, _ = np.histogram2d(df_pairs_py_bg["tof1"], df_pairs_py_bg["tof2"], bins=bins)
    print('python fg')
    print(np.int_(xy_hist_py))
    print('python bg')
    print(np.int_(xy_hist_bg_py))

    # check if lower diagnoal is 0
    assert xy_hist.all() == xy_hist_py.all()
    for i in range(len(xy_hist_bg)):
        if len(xy_hist_bg[i, :i]) > 0:
            assert xy_hist[i, :i].all() == 0, f"{xy_hist[i, :i]}"
            assert xy_hist_bg[i, :i].all() == 0, f"{xy_hist_bg[i, :i]}"
  
def test_pipico_polars_filter_momentum_fixed_cut():
    # df = pl.from_pandas(
    #    pd.DataFrame(pd.read_feather('test_data.feather'), columns=['trigger nr', 'tof', 'px', 'py']))
    # simulate data
    n_bins = 10
    n_shots = 1_000
    n_parts = 10

    tof_min = 4
    tof_max = 4.45
    #df = gen_data(n_shots, n_parts)
    #df = pl.read_parquet("test_data.parquet")
    #da = pl.from_pandas(df)[['trigger nr', 'idx', 'px', 'py', 'pz', 'tof']].to_numpy()
    #df = pl.read_ipc('/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data_abs_momentum_cut.feather')
    df = pd.read_feather('/Users/brombh/data/programm/rust/pipico_simple_example/tests/test_data_abs_momentum_cut.feather')
    df['px'] = df['p_x']
    df['py'] = df['p_y']
    df['pz'] = df['p_z']
    da = pl.from_pandas(df)[['trigger nr', 'idx', 'p_x', 'p_y', 'p_z', 'tof', 'mz']].to_numpy()
    mass_momementum_cut = np.array(
    [[17.5, 18.5, 17.5, 18.5, 1e1**2],
     [16.5, 17.5, 17.5, 18.5, 2.9e1**2],
     [16.5, 17.5, 18.5, 19.5, 9.7e1**2]])
    default_momentum_cut = 3e5**2


    # get pairs from rust
    start = time.time()
    pairs_fg, pairs_bg = pipico.get_covar_pairs_fixed_cut(da, mass_momementum_cut, default_momentum_cut)
    stop = time.time()
    time_rs = stop - start
    print(f"Rust numpy took: {(time_rs)*1e3} ms")
    
    
    #df_pairs_fg, df_pairs_bg = sort_pairs(df, pairs_fg, pairs_bg)
    #assert ((df_pairs_fg['tof2'] - df_pairs_fg['tof1']) >= 0).all(), "TOF1 <= TOF2 needs to be given"
    #assert ((df_pairs_bg['tof2'] - df_pairs_bg['tof1']) >= 0).all(), "TOF1 <= TOF2 needs to be given"


    '''
    import matplotlib.pyplot as plt
    x_hist, x_bins = np.histogram(np.sqrt(
        (df_pairs_fg['px1'] + df_pairs_fg['px2'])**2 + 
        (df_pairs_fg['py1'] + df_pairs_fg['py2'])**2 + 
        (df_pairs_fg['pz1'] + df_pairs_fg['pz2'])**2), bins=100)
    x = (x_bins[:-1] + x_bins[1:]) * 0.5
    plt.plot(x, x_hist)
    plt.show()
    '''

    '''
    bins = np.linspace(df.tof.min(), df.tof.max(), 10)
    xy_hist, _, _ = np.histogram2d(df_pairs_fg["tof1"], df_pairs_fg["tof2"], bins=bins)
    xy_hist_bg, _, _ = np.histogram2d(df_pairs_bg["tof1"], df_pairs_bg["tof2"], bins=bins)
    print('rust fg')
    print(np.int_(xy_hist))
    print('rust bg')
    print(np.int_(xy_hist_bg))

    xy_hist_py, _, _ = np.histogram2d(df_pairs_py_fg["tof1"], df_pairs_py_fg["tof2"], bins=bins)
    xy_hist_bg_py, _, _ = np.histogram2d(df_pairs_py_bg["tof1"], df_pairs_py_bg["tof2"], bins=bins)
    print('python fg')
    print(np.int_(xy_hist_py))
    print('python bg')
    print(np.int_(xy_hist_bg_py))

    # check if lower diagnoal is 0
    assert xy_hist.all() == xy_hist_py.all()
    for i in range(len(xy_hist_bg)):
        if len(xy_hist_bg[i, :i]) > 0:
            assert xy_hist[i, :i].all() == 0, f"{xy_hist[i, :i]}"
            assert xy_hist_bg[i, :i].all() == 0, f"{xy_hist_bg[i, :i]}"
    '''

def sort_pairs(df: pd.DataFrame, pairs: np.array, pairs_bg: np.array) -> (pd.DataFrame, pd.DataFrame):
    # forground
    df_p1 = df.loc[pairs[:, 0]].copy()
    df_p2 = df.loc[pairs[:, 1]].copy()
    df_p1.reset_index(drop=True, inplace=True)
    df_p2.reset_index(drop=True, inplace=True)
    df_p1 = df_p1.add_suffix("1")
    df_p2 = df_p2.add_suffix("2")
    df_pairs = pd.concat([df_p1, df_p2], axis=1)

    # Background
    df_p1 = df.loc[pairs_bg[:, 0]].copy()
    df_p2 = df.loc[pairs_bg[:, 1]].copy()
    df_p1.reset_index(drop=True, inplace=True)
    df_p2.reset_index(drop=True, inplace=True)
    df_p1 = df_p1.add_suffix("1")
    df_p2 = df_p2.add_suffix("2")
    df_pairs_bg = pd.concat([df_p1, df_p2], axis=1)

    return df_pairs, df_pairs_bg


def filter_covariance_py(data: pd.DataFrame, tof_min=0, tof_max=10, n_bins=10):
    # define function to be run on thread
    bins = np.linspace(tof_min, tof_max, n_bins+1)

    data.sort_values(['trigger nr', 'tof'], inplace=True)
        
    hist2d    = np.zeros((n_bins, n_bins))
    hist2d_bg = np.zeros((n_bins, n_bins))
    pairs = []
    pairs_bg = []

    for trigger_nr, df in data.groupby(["trigger nr"])['tof', 'px', 'py', 'pz', 'idx']:
        row_tof = df['tof'].to_numpy()
        row_px  = df['px'].to_numpy()
        row_py  = df['py'].to_numpy()
        row_pz  = df['pz'].to_numpy()
        row_idx = df['idx'].to_numpy()

        # get data for bg computation
        df_bg = data.query(f"`trigger nr` != {trigger_nr}").sample(len(row_tof))
        row_tof_bg = df_bg['tof'].to_numpy()
        row_px_bg  = df_bg['px'].to_numpy()
        row_py_bg  = df_bg['py'].to_numpy()
        row_pz_bg  = df_bg['pz'].to_numpy()
        row_idx_bg = df_bg.index.to_numpy()
        for p1, tof in enumerate(row_tof):  # go through every element in the row
            idx_x = np.digitize(tof, bins = bins) - 1
            p2 = p1 + 1
            px = row_px[p1]
            py = row_py[p1]
            pz = row_pz[p1]
            mask = (row_px[p2:] + px)**2 + (row_py[p2:] + py)**2 + (row_pz[p2:] + pz)**2 <= (px**2 + py**2 + pz**2) * 0.0025
            row = row_tof[p2:][mask]
            for y, idx in zip(row, row_idx[p2:][mask]):
                # idx_y = np.digitize(row_tof[p2], bins=bins) - 1
                idx_y = np.digitize(y, bins = bins) - 1
                hist2d[idx_y, idx_x] += 1
                #indizes.append(row_idx[p1])
                pairs.append([row_idx[p1], idx])
            #[indizes.append(i) for i in row_idx[p2:][mask]]


            # calc background
            mask = (row_px_bg[p2:] + px)**2 + (row_py_bg[p2:] + py)**2 + (row_pz_bg[p2:] + pz)**2 <= (px**2 + py**2 + pz**2) * 0.0025
            row_bg = row_tof_bg[p2:][mask]
            for y, idx in zip(row_bg, row_idx_bg[p2:][mask]):
                # idx_y = np.digitize(row_tof[p2], bins=bins) - 1
                hist_idx_y = np.digitize(y, bins = bins) - 1
                if tof <= y:
                    hist2d_bg[hist_idx_y, idx_x] += 1
                    pairs_bg.append([row_idx[p1], idx])                    
                else:
                    hist2d_bg[idx_x, hist_idx_y] += 1
                    pairs_bg.append([idx, row_idx[p1]])
                    
    return hist2d, hist2d_bg, pairs, pairs_bg


def gen_data(Ntr=100, n_parts=10):
    def ptot(event):
        # calculate the time of flight
        p = event[4]
        m = event[5]
        c_1 = 1
        t_0 = c_1 * np.sqrt(m)
        c_2 = 0.01  # to scale the size of the PIPICO line
        t = t_0 + p * c_2
        return t

    def detection(A, tr, Nf, p_x, p_y, p_z, m):
        t = ptot([tr, Nf, p_x, p_y, p_z, m])
        if len(A) > 0:
            A = np.vstack((A, np.array([tr, Nf, p_x, p_y, p_z, m, t])))
        else:
            A = np.array([[tr, Nf, p_x, p_y, p_z, m, t]])
        return A

    def single_trigger(Nf, tr, n_parts = 10):
        A = []
        for _ in range(n_parts): # generate data for n particles
            nf = int((np.random.poisson(Nf)))
            for f in range(nf):
                E_tot = 1
                channel = int(np.random.uniform(0, 3))
                if channel == 0:
                    m_1 = 18
                    m_2 = 18
                elif channel == 1:
                    m_1 = 17
                    m_2 = 18
                else:
                    m_1 = 17
                    m_2 = 19
                mu = m_1 * m_2 / (m_1 + m_2)
                p = np.sqrt(2 * E_tot * mu)

                # uniform distribution
                cost = np.random.uniform(-1, 1)
                sint = np.sqrt(1 - cost**2)
                phi = np.random.uniform(0, 2 * np.pi)
                cosp = np.cos(phi)
                sinp = np.sin(phi)

                # fill momenta
                p_x = p * cosp * sint
                p_y = p * sinp * sint
                p_z = p * cost
                alpha = 0.5

                # fill first particle
                if np.random.uniform(0, 1) < alpha:
                    A = detection(A, tr, Nf, p_x, p_y, p_z, m_1)

                # fill second particle
                if np.random.uniform(0, 1) < alpha:
                    A = detection(A, tr, Nf, -p_x, -p_y, -p_z, m_2)

        if np.ndim(A) > 0 and len(A) > 0:
            A = A[A[:, 6].argsort()]
        
        return A


    Nf = 3  # number of fragmentation events
    # create a data set
    A = []
    tr = 0
    while tr < Ntr:
        tr_sub = 0
        A_sub = []
        first = True
        while tr_sub < 1000 and tr < Ntr:  # divide the Montecarlo sampling
            # into smaller steps to speed up the process.
            Atr = single_trigger(Nf, tr, n_parts)  # single trigger data
            tr += 1
            tr_sub += 1
            if first and len(Atr) != 0:
                A_sub = Atr
                first = False
            elif len(Atr) != 0:
                A_sub = np.vstack((A_sub, Atr))
            else:
                continue
        if len(A) > 0:  # A has some entries already
            A = np.vstack((A, A_sub))
        else:  # A is empty
            A = A_sub

    print("\n")
    # A = tr, Nf, p_x, p_y, p_z, m, t
    df = pd.DataFrame(A, columns=["trigger nr", "num part", "px", "py", "pz", "m", "tof"])
    df.sort_values(['trigger nr', 'tof'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['idx'] = df.index
    df.to_parquet("test_data.parquet")
    Nions = np.shape(A)[0]
    print("Total number of ions:", Nions)
    print("Number of ions with m=17:", np.shape(A[A[:, 5] == 17])[0])
    print("Number of ions with m=18:", np.shape(A[A[:, 5] == 18])[0])
    print("Number of ions with m=19:", np.shape(A[A[:, 5] == 19])[0])


    return df


def test_pipico_polars_filter_momentum_simple():
    """simple test to simulate the order of iteration"""
    # generate data
    n_shots = 3
    n_parts = 4
    data = np.array([[i for i in range(n_parts)] for j in range(n_shots)], dtype=np.float64)

    print(data)

    trigger_nrs = [num for vec in [len(row) * [i] for i, row in enumerate(data)] for num in vec]
    x = [i for vec in data for i in vec]
    df = pl.from_pandas(
        pd.DataFrame(np.column_stack((trigger_nrs, x, x)), columns=["trigger nr", "mz", "p_abs"])
    )

    # this is what it should be
    print("python")
    for i, row in enumerate(data):
        p1 = 0
        p2 = 0
        print()
        while p1 < len(row):
            p2 = p1 + 1
            while p2 < len(row):
                # print((i, p1), (i, p2))
                print(f"{p1}={row[p1]}, {p2}={row[p2]}")
                p2 += 1
            p1 += 1

    print("rust")
    hist = pipico.polars_filter_momentum_pl(
        pydf=df[["trigger nr", "mz", "p_abs"]],
        col_grp="trigger nr",
        col_pipico="mz",
        col_mask="p_abs",
        filter_delta=10,
        n_bins=10,
        hist_min=1,
        hist_max=6,
    )


if __name__ == "__main__":
    #test_pipico_polars_filter_momentum_simulated()
    test_pipico_polars_filter_momentum_fixed_cut()
    # test_pipico_polars_filter_momentum()
    # test_pipico_polars_filter_momentum_simple()