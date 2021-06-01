import pathlib
import time

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
try:
    import tqdm
except:
    tqdm = None

import ckmeans

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

    k = 10
    n_rep = 100
    p_feat = 0.5
    p_samp = 0.5

    
    ckm_0 = ckmeans.CKmeans(
        k=k,
        n_rep=n_rep,
        p_samp=p_samp,
        p_feat=p_feat,
        metrics=[
            'sil',
            'bic',
            'db',
            'ch',
        ]
    )

    print('fitting ...')
    if tqdm:
        with tqdm.tqdm(total=n_rep) as bar:
            t0 = time.time()
            ckm_0.fit(x_0, progress_callback=bar.update)
            t1 = time.time()
    else:
        t0 = time.time()
        ckm_0.fit(x_0)
        t1 = time.time()

    print('predicting ...')
    if tqdm:
        with tqdm.tqdm(total=n_rep) as bar:
            t2 = time.time()
            ckm_0_res = ckm_0.predict(x_0, progress_callback=bar.update)
            t3 = time.time()
    else:
        t2 = time.time()
        ckm_0_res = ckm_0.predict(x_0)
        t3 = time.time()

    print(ckm_0_res.cmatrix)

    print(t1 - t0)
    print(t3 - t2)

    fig, ax = plt.subplots(1,1)
    ax.imshow(ckm_0_res.sort().cmatrix)
    fig.savefig(path / 'manual_test_img0.png')

    print('sils:', ckm_0.sils)
    print('bics:', ckm_0.bics)
    print('dbs:', ckm_0.dbs)
    print('chs:', ckm_0.chs)
