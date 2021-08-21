import pathlib
import time

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

import pyckmeans

if __name__ == '__main__':

    path = pathlib.Path(__file__).parent.absolute()

    p = 10

    n0 = 50
    x0 = np.random.normal(0, 2, (n0, p))
    n1 = 50
    x1 = np.random.normal(-3, 1.5, (n1, p))
    n2 = 50
    x2 = np.random.normal(3, 2, (n2, p))

    x_0 = np.r_[x0, x1, x2]

    k = np.arange(2, 8)
    n_rep = 200
    p_feat = 0.8
    p_samp = 0.8
    gamma = 0.5
    must_link = np.array([
        [0, 10],
        [12, 21],
        [52, 56],
        [75, 61],
        [101, 142],
        # [1, 51],
        # [2, 51],
        # [3, 51],
        # [4, 51],
        # [5, 51],
        # [6, 51],
        # [7, 51],
        # [8, 51],
        # [9, 51],
        # [10, 51]
    ])
    must_not_link = np.array([
        [0, 64],
        [88, 15],
        [112, 56],
        [140, 1],
        # [1, 2],
        # [1, 3],
        # [1, 4],
        # [1, 5],
        # [1, 6],
        # [1, 7],
    ])

    wecr_0 = pyckmeans.WECR(k=k, n_rep=n_rep, p_samp=p_samp, p_feat=p_feat, gamma=gamma)

    t0 = time.time()
    wecr_0.fit(x_0, must_link=must_link, must_not_link=must_not_link)
    t1 = time.time()


    t2 = time.time()
    cmatrix = wecr_0.predict(x_0)
    t3 = time.time()

    # print(cmatrix)

    print(t1 - t0)
    print(t3 - t2)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(cmatrix)
    fig.savefig(path / 'manual_test_2_img0.png')

    print(cmatrix)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_0[:, 0], x_0[:, 1])
    fig.savefig(path / 'manual_test_2_img1.png')

    # print(wecr_0.qualities)