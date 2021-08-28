# pyckmeans

pyckmeans is a Python package for [Consensus K-Means clustering](https://doi.org/10.1023/A:1023949509487), especially in the context of DNA sequence data. In addition to the clustering functionality, it provides tools for working with DNA sequence data such as reading and writing of DNA alignment files, calculating genetic distances, and Principle Coordinate Analysis (PCoA) for dimensionality reduction.

## Documentation
See pyckmeans' [Documentation](https://pyckmeans.readthedocs.io/) for details.

## Installation

pyckmeans can be installed using pip, Conda, or from source.

### pip

```bash
pip install pyckmeans
```

### Conda

```bash
conda install pyckmeans -c TankredO
```

### From Source

The installation from source requires `git` and a c++ compiler.

```bash
git clone https://github.com/TankredO/pyckmeans
cd pyckmeans
pip install .
```

## Getting Started

### Clustering a Data Matrix (Single K)

```python
from pyckmeans import CKmeans

# simulate dataset
# 50 samples, 2 features, 3 true clusters
import sklearn.datasets
x, _ = sklearn.datasets.make_blobs(n_samples=50, n_features=2, centers=3, random_state=75)

# apply Consensus K-Means
# 3 clusters, 100 K-Means runs,
# draw 80% of samples and 50% of features for each single K-Means
ckm = CKmeans(k=3, n_rep=100, p_samp=0.8, p_feat=0.5)
ckm.fit(x)
ckm_res = ckm.predict(x)

# plot consensus matrix and consensus clustering
ckm_res.plot(figsize=(5,5))

# consensus matrix
ckm_res.cmatrix

# clustering metrics
print('Bayesian Information Criterion:', ckm_res.bic)
print('Davies-Bouldin Index:', ckm_res.db)
print('Silhouette Score:', ckm_res.sil)
print('Calinski-Harabasz Index:', ckm_res.ch)

# consensus clusters
print('Cluster Membership:', ckm_res.cl)
```

### Clustering a Data Matrix (Multi K)

```python
from pyckmeans import MultiCKMeans
import sklearn.datasets

# simulate dataset
# 50 samples, 10 features, 3 true clusters
x, _ = sklearn.datasets.make_blobs(n_samples=50, n_features=10, centers=3, random_state=44)

# apply multiple Consensus K-Means for
# k = 2, ..., 5
# 100 K-Means runs per Consensus K-Means
# draw 80% of the sample for each single K-Means
# draw 50% of the features for each single K-Means
mckm = MultiCKMeans(k=[2, 3, 4, 5], n_rep=100, p_samp=0.8, p_feat=0.5)
mckm.fit(x)
mckm_res = mckm.predict(x)

# clustering metrics
print('Metrics:')
print(mckm_res.metrics)

# plot clustering metrics against k
# BIC, DB: lower is better
# SIL, CH: higher is better
mckm_res.plot_metrics(figsize=(10,5))


# get a single CKmeansResult                  0 |1| 2  3
ckm_res_k3 = mckm_res.ckmeans_results[1] # k=[2, 3, 4, 5]
ckm_res_k3.plot()
# ...
# see "Clustering a Data Matrix (Single K)"
```

More coming soon.