import pipico

import numpy as np


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
