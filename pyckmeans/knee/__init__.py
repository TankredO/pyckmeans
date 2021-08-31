''' Knee and elbow search.
'''

from typing import Callable, Iterable
import warnings

import numpy

def rel_extrema_idcs(
    x: numpy.ndarray,
    cmp_fun: Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray] = numpy.greater,
    mode: str = 'clip',
) -> numpy.ndarray:
    '''rel_extrema_idcs

    Find indices of relative extrema. A relative extremum is found if
    at an element, if cmp_fun returns true for both of its neighbors.

    Parameters
    ----------
    x : numpy.ndarray
        Data vector.
    cmp_fun : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray], optional
        Compare function function, by default numpy.greater
    mode : str, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' - raise an error (default)
        * 'wrap' - wrap around
        * 'clip' - clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

        (mode documentation copied from numpy.take)

    Returns
    -------
    numpy.ndarray
        Indices of the extrema.
    '''
    idcs = numpy.arange(0, x.shape[0])
    left = x.take(idcs + 1, mode=mode)
    right = x.take(idcs - 1, mode=mode)

    return numpy.nonzero(cmp_fun(x, left) & cmp_fun(x, right))[0]


# the following code is mostly copied from
# https://github.com/arvkevi/kneed and
# was adapted for compatibility

VALID_CURVE = ('convex', 'concave')
VALID_DIRECTION = ('increasing', 'decreasing')

class KneeLocator(object):
    '''KneeLocator

    An implementation of the Kneedle algorithm [1]_.

    Once instantiated, this class attempts to find the point of maximum
    curvature on a line. The knee is accessible via the `.knee` attribute.

    Parameters
    ----------
    x : numpy.ndarray
        x values.
    y : numpy.ndarray
        y values.
    S : float, optional
        Sensitivity, original paper suggests default of 1.0, by default 1.0.
    curve : str, optional
        If 'concave', algorithm will detect knees. If 'convex', it
        will detect elbows., by default 'concave'.
    direction : str, optional
        Curve direction. One of {'increasing', 'decreasing'}, by default 'increasing'.
    interp_method : str, optional
        Interpolation method. One of

        * 'interp1d' - no interpolation
        * 'polynomial' - polynomial interpolation

        By default 'interp1d'.
    online : bool, optional
        Correct old knee points if True, will return first knee if False,
        by default False.
    polynomial_degree : int, optional
        The degree of the fitting polynomial. Only used when interp_method='polynomial'.
        This argument is passed to numpy polyfit `deg` parameter., by default 7.

    Raises
    ------
    ValueError
        Raised if invalid curve or direction argument passed.
    ValueError
        Raised if invalid interp_method argument passed.

    References
    ----------
    .. [1]  Satopaa, V., J., Albrecht, D., Irwin, B., Raghavan. 2011.
            "Finding a "Kneedle" in a Haystack: Detecting Knee Points in System Behavior".
            31st International Conference on Distributed Computing Systems Workshops.
            doi: 10.1109/ICDCSW.2011.20.
    '''

    def __init__(
        self,
        x: numpy.ndarray,
        y: numpy.ndarray,
        S: float = 1.0,
        curve: str = 'concave',
        direction: str = 'increasing',
        interp_method: str = 'interp1d',
        online: bool = False,
        polynomial_degree: int = 7,
    ):
        # Step 0: Raw Input
        self.x = numpy.array(x)
        self.y = numpy.array(y)
        self.curve = curve
        self.direction = direction
        self.N = len(self.x)
        self.S = S
        self.all_knees = set()
        self.all_norm_knees = set()
        self.all_knees_y = []
        self.all_norm_knees_y = []
        self.online = online
        self.polynomial_degree = polynomial_degree

        valid_curve = self.curve in VALID_CURVE
        valid_direction = self.direction in VALID_DIRECTION
        if not all((valid_curve, valid_direction)):
            raise ValueError(
                'Please check that the curve and direction arguments are valid.'
            )

        # Step 1: fit a smooth line
        if interp_method == 'interp1d':
            # uspline = interpolate.interp1d(self.x, self.y)
            # self.ds_y = uspline(self.x)
            self.ds_y = y
        elif interp_method == 'polynomial':
            p = numpy.poly1d(numpy.polyfit(x, y, self.polynomial_degree))
            self.ds_y = p(x)
        else:
            msg = f'{interp_method} is an invalid interp_method parameter, ' +\
                'use either "interp1d" or "polynomial".'
            raise ValueError(msg)

        # Step 2: normalize values
        self.x_normalized = self._normalize(self.x)
        self.y_normalized = self._normalize(self.ds_y)

        # Step 3: Calculate the Difference curve
        self.y_normalized = self.transform_y(
            self.y_normalized, self.direction, self.curve
        )
        # normalized difference curve
        self.y_difference = self.y_normalized - self.x_normalized
        self.x_difference = self.x_normalized.copy()

        # Step 4: Identify local maxima/minima
        # local maxima
        self.maxima_indices = rel_extrema_idcs(self.y_difference, numpy.greater_equal)
        self.x_difference_maxima = self.x_difference[self.maxima_indices]
        self.y_difference_maxima = self.y_difference[self.maxima_indices]

        # local minima
        self.minima_indices = rel_extrema_idcs(self.y_difference, numpy.less_equal)
        self.x_difference_minima = self.x_difference[self.minima_indices]
        self.y_difference_minima = self.y_difference[self.minima_indices]

        # Step 5: Calculate thresholds
        self.Tmx = self.y_difference_maxima - (
            self.S * numpy.abs(numpy.diff(self.x_normalized).mean())
        )

        # Step 6: find knee
        self.knee, self.norm_knee = self.find_knee()

        # Step 7: If we have a knee, extract data about it
        self.knee_y = self.norm_knee_y = None
        if self.knee:
            self.knee_y = self.y[self.x == self.knee][0]
            self.norm_knee_y = self.y_normalized[self.x_normalized == self.norm_knee][0]

    @staticmethod
    def _normalize(x: numpy.ndarray) -> numpy.ndarray:
        '''_normalize

        Scale vector values between 0 and 1.

        Parameters
        ----------
        x : numpy.ndarray
            Vector to scale.

        Returns
        -------
        numpy.ndarray
            Scaled vector
        '''
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def transform_y(y: Iterable[float], direction: str, curve: str) -> float:
        '''transform y to concave, increasing based on given direction and curve'''
        # convert elbows to knees
        if direction == 'decreasing':
            if curve == 'concave':
                y = numpy.flip(y)
            elif curve == 'convex':
                y = y.max() - y
        elif direction == 'increasing' and curve == 'convex':
            y = numpy.flip(y.max() - y)

        return y

    def find_knee(self,):
        '''
        This function is called when KneeLocator is instantiated.
        It identifies the knee value and sets the instance attributes.
        '''
        if not self.maxima_indices.size:
            warnings.warn(
                'No local maxima found in the difference curve\n'
                'The line is probably not polynomial.',
                RuntimeWarning,
            )
            return None, None
        # placeholder for which threshold region i is located in.
        maxima_threshold_index = 0
        minima_threshold_index = 0
        # traverse the difference curve
        for i, x in enumerate(self.x_difference):
            # skip points on the curve before the the first local maxima
            if i < self.maxima_indices[0]:
                continue

            j = i + 1

            # reached the end of the curve
            if x == 1.0:
                break

            # if we're at a local max, increment the maxima threshold index and continue
            if (self.maxima_indices == i).any():
                threshold = self.Tmx[maxima_threshold_index]
                threshold_index = i
                maxima_threshold_index += 1
            # values in difference curve are at or after a local minimum
            if (self.minima_indices == i).any():
                threshold = 0.0
                minima_threshold_index += 1

            if self.y_difference[j] < threshold:
                if self.curve == 'convex':
                    if self.direction == 'decreasing':
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]

                elif self.curve == 'concave':
                    if self.direction == 'decreasing':
                        knee = self.x[-(threshold_index + 1)]
                        norm_knee = self.x_normalized[threshold_index]
                    else:
                        knee = self.x[threshold_index]
                        norm_knee = self.x_normalized[threshold_index]

                # add the y value at the knee
                y_at_knee = self.y[self.x == knee][0]
                y_norm_at_knee = self.y_normalized[self.x_normalized == norm_knee][0]
                if knee not in self.all_knees:
                    self.all_knees_y.append(y_at_knee)
                    self.all_norm_knees_y.append(y_norm_at_knee)

                # now add the knee
                self.all_knees.add(knee)
                self.all_norm_knees.add(norm_knee)

                # if detecting in offline mode, return the first knee found
                if self.online is False:
                    return knee, norm_knee

        if self.all_knees == set():
            warnings.warn('No knee/elbow found')
            return None, None

        return knee, norm_knee

    # Niceties for users working with elbows rather than knees
    @property
    def elbow(self):
        return self.knee

    @property
    def norm_elbow(self):
        return self.norm_knee

    @property
    def elbow_y(self):
        return self.knee_y

    @property
    def norm_elbow_y(self):
        return self.norm_knee_y

    @property
    def all_elbows(self):
        return self.all_knees

    @property
    def all_norm_elbows(self):
        return self.all_norm_knees

    @property
    def all_elbows_y(self):
        return self.all_knees_y

    @property
    def all_norm_elbows_y(self):
        return self.all_norm_knees_y
