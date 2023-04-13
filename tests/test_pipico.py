import time
import numpy as np
import pandas as pd
import polars as pl

import pipico


def test_pipico_equal_size():
    data = [
        [2.5, 3.2, 3.2, 3.2, 3.2, 3.4, 3.5, 3.6, 4.1, 4.9],
        [1.5, 3.2, 3.2, 3.2, 3.2, 3.3, 3.3, 3.4, 4.1, 4.2],
        [1.5, 1.5, 1.5, 1.5, 1.5, 3.1, 3.2, 3.2, 3.6, 3.6],
        [1.5, 1.5, 1.5, 1.5, 2.5, 3.4, 3.5, 3.5, 4.1, 4.9],
        [2.5, 3.2, 3.2, 3.2, 3.5, 3.6, 4.1, 5.5, 5.5, 5.5],
        [1.5, 1.5, 3.2, 3.2, 3.2, 3.2, 3.5, 3.5, 4.1, 4.1],
        [1.5, 3.2, 3.4, 3.5, 3.6, 3.7, 4.1, 4.1, 4.1, 4.1],
        [1.5, 1.5, 1.5, 1.6, 3.1, 3.2, 3.2, 3.2, 3.4, 4.9],
        [1.5, 3.2, 3.2, 3.2, 3.2, 3.4, 3.5, 4.1, 4.2, 5.5],
        [1.5, 1.5, 1.5, 1.5, 1.5, 2.6, 3.2, 3.2, 3.2, 4.1],
    ]
    should_be = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 33.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 76.0, 0.0, 12.0, 67.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 26.0, 0.0, 6.0, 43.0, 8.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 21.0, 0.0, 4.0, 52.0, 24.0, 9.0, 0.0, 0.0, 0.0],
            [0.0, 8.0, 0.0, 2.0, 11.0, 4.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 3.0, 14.0, 7.0, 5.0, 0.0, 0.0, 3.0],
        ]
    )
    hist = pipico.pipico_equal_size(np.asarray(data), 10, 1, 6)

    # also check that the two implementation yield the same results
    hist1 = pipico.pipico_lists(data, 10, 1, 6)

    assert (hist == should_be).all()
    assert (hist == hist1).all()


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


def test_pipico_polars_filter_momentum_simulated():
    # df = pl.from_pandas(
    #    pd.DataFrame(pd.read_feather('test_data.feather'), columns=['trigger nr', 'tof', 'px', 'py']))
    # simulate data
    n_bins = 100
    n_shots = 1_000
    n_parts = 10

    tof_min = 0
    tof_max = 10
    df, data_tof, data_px, data_py = gen_data(n_shots, n_parts)
    #df = pl.read_parquet("test_data.parquet")
    df = pl.from_pandas(df)
    da = df[['trigger nr', 'idx', 'px', 'py']].to_numpy()

    '''
    start = time.time()
    hist_pl = pipico.polars_filter_momentum_pl(
        pydf=df[["trigger nr", "tof", "px", "py"]],
        col_grp="trigger nr",
        col_pipico="tof",
        col_mask1="px",
        col_mask2="py",
        filter_delta=0.01,
        n_bins=n_bins,
        hist_min=tof_min,
        hist_max=tof_max,
    )
    stop = time.time()
    #print(f"Rust polars took: {stop - start} s")
    #print("Rust")
    #print(hist_pl)
    '''

    start = time.time()
    hist_np = pipico.polars_filter_momentum_np(
        x=da, filter_delta=0.01, n_bins=n_bins, hist_min=tof_min, hist_max=tof_max
    )
    stop = time.time()
    print(f"Rust numpy took: {(stop - start)*1e3} ms")
    print(hist_np)

    start = time.time()
    hist_py = filter_cov_py(data_tof, data_px, data_py, n_bins, tof_min, tof_max)
    stop = time.time()
    print(f"Python took: {(stop - start)*1e3} ms")
    print(hist_py)

    #print("difference:")
    #print(hist_pl - hist_py)
    #print(hist_np - hist_py)
    assert hist_np.all() == hist_py.all()


def filter_cov_py(data_tof, data_px, data_py, n_bins, tof_min, tof_max):
    hist2d = np.zeros((n_bins, n_bins))
    bins = np.linspace(tof_min, tof_max, n_bins + 1)
    for row_tof, row_px, row_py in zip(data_tof, data_px, data_py):
        idx = row_tof.argsort()
        row_tof = row_tof[idx]
        row_px = row_px[idx]
        row_py = row_py[idx]
        for p1, tof in enumerate(row_tof):  # go through every element in the row
            idx_x = np.digitize(tof, bins=bins) - 1
            p2 = p1 + 1
            px = row_px[p1]
            py = row_py[p1]
            #row = row_tof[p2:][((row_px[p2:] + px) ** 2 < 0.01) & ((row_py[p2:] + py) ** 2 < 0.01)]
            row = row_tof[p2:][(row_px[p2:] + px)**2 + (row_py[p2:] + py)**2 < (px**2 + py**2)*0.0025]
            for y in row:
                # idx_y = np.digitize(row_tof[p2], bins=bins) - 1
                idx_y = np.digitize(y, bins=bins) - 1
                hist2d[idx_y, idx_x] += 1
    return hist2d


def gen_data(n_shots=100, n_parts=10):
    # simulate some data
    data_tof = np.zeros((n_shots, n_parts))
    for i in range(n_shots):
        dt1 = np.random.uniform(-0.25, 0.25)
        data_tof[i][0] = 4 - dt1
        data_tof[i][1] = 2 + dt1
        data_tof[i][2] = 6 - dt1
        data_tof[i][3] = 8 + dt1
        data_tof[i][4:] = np.random.uniform(0, 10, n_parts - 4)

    data_px = np.zeros((n_shots, n_parts))
    data_py = np.zeros((n_shots, n_parts))
    for i in range(n_shots):
        dt = np.random.uniform(-10, 10, 2)
        data_px[i][0] = dt[0]
        data_py[i][0] = dt[0]
        data_px[i][1] = -dt[0]
        data_py[i][1] = -dt[0]
        data_px[i][2] = dt[1]
        data_py[i][2] = dt[1]
        data_px[i][3] = -dt[1]
        data_py[i][3] = -dt[1]

        data_px[i][4:] = np.random.uniform(-10, 10, n_parts - 4)
        data_py[i][4:] = np.random.uniform(-10, 10, n_parts - 4)

    # convert list to DataFrame
    trigger_nrs = [
        num for vec in [len(row) * [i] for i, row in enumerate(data_tof)] for num in vec
    ]
    data_tof_col = [i for row in data_tof for i in row]
    data_px_col = [i for row in data_px for i in row]
    data_py_col = [i for row in data_py for i in row]

    df = pd.DataFrame(
        np.column_stack((trigger_nrs, data_tof_col, data_px_col, data_py_col)),
        columns=("trigger nr", "tof", "px", "py"),
    )
    df.sort_values(['trigger nr', 'tof'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    df['idx'] = df.index

    df.to_parquet("test_data.parquet")
    return df, data_tof, data_px, data_py


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
    test_pipico_polars_filter_momentum_simulated()
    # test_pipico_polars_filter_momentum()
    # test_pipico_polars_filter_momentum_simple()