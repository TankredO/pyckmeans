''' Progress bar utilities
'''

from typing import Dict, Any

import tqdm
from pyckmeans.core import MultiCKMeans

class MultiCKMeansProgressBars:
    '''MultiCKMeansProgressBars

    Context Manager for a MultiCKMeans progress bars.

    Parameters
    ----------
    mckm : MultiCKMeans
        MultiCKMeans object to display progress bars for.
    kwargs : Dict[str, Any]
        Additional keyword arguments passed to tqdm.tqdm.
    '''
    def __init__(
        self,
        mckm: MultiCKMeans,
        **kwargs: Dict[str, Any],
    ):
        self.mckm = mckm

        self.ks = mckm.k
        self.n_rep = mckm.n_rep

        self._ckm_idx = 0
        self._iter = 0
        self._done = False

        # tqdm options
        self._tqdm_kwargs = {
        }
        self._tqdm_kwargs.update(kwargs)

        # init first progress bar
        self._tqdm = tqdm.tqdm(
            total=self.n_rep,
            mininterval=0.5,
            desc=f'k={self.ks[self._ckm_idx]}',
            **self._tqdm_kwargs,
        )

    def update(
        self,
        n: int = 1,
    ):
        '''update

        Update progress by n iterations.

        Parameters
        ----------
        n : int, optional
            Progress increment in iterations, by default 1
        '''
        if self._done:
            return

        self._iter += n
        self._tqdm.update(n)

        if self._iter >= self.n_rep:
            self._tqdm.close()
            self._iter = 0
            self._ckm_idx += 1
            if self._ckm_idx >= len(self.ks):
                self._done = True
            else:
                self._tqdm = tqdm.tqdm(
                    total=self.n_rep,
                    desc=f'k={self.ks[self._ckm_idx]}',
                    **self._tqdm_kwargs,
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._tqdm.close()
        return
