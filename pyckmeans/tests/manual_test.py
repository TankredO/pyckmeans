import pathlib
import time
import pickle

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
try:
    import tqdm
except:
    tqdm = None

import pyckmeans
import pyckmeans.utils

if __name__ == '__main__':
    path = pathlib.Path(__file__).parent.absolute()

    p = 10

    n0 = 50
    x0 = np.random.normal(0, 2, (n0, p))
    n1 = 50
    x1 = np.random.normal(-5, 1.5, (n1, p))
    n2 = 50
    x2 = np.random.normal(5, 2, (n2, p))

    x_0 = np.r_[x0, x1, x2]

    k = 3
    n_rep = 100
    p_feat = 0.5
    p_samp = 0.5

    
    ckm_0 = pyckmeans.CKmeans(
        k=k,
        n_rep=n_rep,
        p_samp=p_samp,
        p_feat=p_feat,
        metrics=[
            'sil',
            'bic',
            'db',
            'ch',
        ],
        n_init=5,
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
            ckm_0_res = ckm_0.predict(x_0, progress_callback=bar.update, return_cls=True)
            t3 = time.time()
    else:
        t2 = time.time()
        ckm_0_res = ckm_0.predict(x_0, return_cls=True)
        t3 = time.time()

    print(ckm_0_res.cmatrix)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_00.tsv', one_hot=False, row_names=False, col_names=False)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_10.tsv', one_hot=False, row_names=True, col_names=False)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_01.tsv', one_hot=False, row_names=False, col_names=True)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_11.tsv', one_hot=False, row_names=True, col_names=True)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_oh_00.tsv', one_hot=True, row_names=False, col_names=False)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_oh_10.tsv', one_hot=True, row_names=True, col_names=False)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_oh_01.tsv', one_hot=True, row_names=False, col_names=True)
    ckm_0_res.save_km_cls(path / 'ckm_0_res_km_cls_oh_11.tsv', one_hot=True, row_names=True, col_names=True)

    print(t1 - t0)
    print(t3 - t2)

    # fig, ax = plt.subplots(1,1)
    # ax.imshow(ckm_0_res.sort().cmatrix)
    # fig.savefig(path / 'manual_test_img0.png')

    fig = ckm_0_res.plot(figsize=(10, 10))
    fig.savefig(path / 'manual_test_img0.png')

    fig = ckm_0_res.plot(
        names=np.arange(x_0.shape[0]).astype('str'),
        figsize=(10, 10),
    )
    fig.savefig(path / 'manual_test_img1.png')

    fig = ckm_0_res.plot(
        names=np.arange(x_0.shape[0]).astype('str'),
        figsize=(10, 10),
        order=None,
    )
    fig.savefig(path / 'manual_test_img2.png')

    print('sils:', ckm_0.sils)
    print('bics:', ckm_0.bics)
    print('dbs:', ckm_0.dbs)
    print('chs:', ckm_0.chs)

    ks = [2,3,4,5,6,7,8,9,10]
    n_rep = 100
    mckm_0 = pyckmeans.MultiCKMeans(k=ks, n_rep=n_rep)

    print('fitting multi ...')
    with pyckmeans.utils.MultiCKMeansProgressBars(mckm_0) as pb:
        mckm_0.fit(x_0, pb.update)

    with open(path / 'mckm_0.pickle', 'wb') as f:
        pickle.dump(mckm_0, f)

    print('predicting multi ...')
    with pyckmeans.utils.MultiCKMeansProgressBars(mckm_0) as pb:
        mckm_0_res = mckm_0.predict(x_0, progress_callback=pb.update)

    print('sils:', mckm_0_res.sils)
    print('bics:', mckm_0_res.bics)
    print('dbs:', mckm_0_res.dbs)
    print('chs:', mckm_0_res.chs)

    fig = mckm_0_res.plot_metrics(figsize=(10, 10))
    fig.savefig(path / f'manual_test_img_metrics0.png')

    for k, ckm_res in zip(ks, mckm_0_res.ckmeans_results):
        fig = ckm_res.plot(figsize=(10, 10))
        fig.savefig(path / f'manual_test_img_k-{k}.png')
