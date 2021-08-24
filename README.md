# pyckmeans

pyckmeans is a Python package for [Consensus K-Means clustering](https://doi.org/10.1023/A:1023949509487) in the context of genetic sequence data. In addition to the clustering functionality, it provides tools for working with DNA sequence data such as reading and writing of DNA alignment files, calculating genetic distances, and Principle Coordinate Analysis (PCoA) for dimensionality reduction of the latter.

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

```python
from pyckmeans import CKmeans
from pyckmeans.io import NucleotideAlignment
from pyckmeans.ordination import pcoa
```

More coming soon.