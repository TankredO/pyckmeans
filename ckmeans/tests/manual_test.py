import pathlib
import time

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

import ckmeans

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

ckm_0 = ckmeans.CKmeans(k, n_rep)

t0 = time.time()
ckm_0.fit(x_0)
t1 = time.time()


t2 = time.time()
cmatrix = ckm_0.predict(x_0)
t3 = time.time()

print(cmatrix)

print(t1 - t0)
print(t3 - t2)

fig, ax = plt.subplots(1,1)
ax.imshow(cmatrix)
fig.savefig(path / 'test.png')

print(ckm_0.sils)