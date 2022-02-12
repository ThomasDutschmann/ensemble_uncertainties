
"""Library of custom kernel functions for SVM."""

import numpy as np

from sklearn.metrics import pairwise_distances


def tanimoto(A, B):
    """Fast implementation of the Tanimoto similarity.

    Parameters
    ----------
    A : list-like
        The first bit-sequence
    B : list-like
        The second bit-sequence
    
    Returns
    -------
    float
        Tanimoto similarity (\in [0, 1])
    """
    def metric(a, b):
        # Check where both entries have a 1
        ab = np.sum((a + b) == 2)
        return ab / (np.sum(a) + np.sum(b) - ab)
    return pairwise_distances(A, B, metric=metric)
