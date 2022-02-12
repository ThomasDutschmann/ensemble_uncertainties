
"""Test self-implemented custom kernel functions."""

import unittest

import numpy as np

from ensemble_uncertainties.utils.kernel_functions import (
    tanimoto
)


class KernelFunctionTests(unittest.TestCase):

    def test_tanimoto_kernel(self):
        """Evaluate Tanimoto similarity manually."""
        # Similarity to itself should be 1
        set1 = np.array([[1, 1, 1, 1, 1, 1]])
        self.assertEqual(tanimoto(set1, set1), 1.0)
        # Similarity to set1 should be 0
        set2 = np.array([[0, 0, 0, 0, 0, 0]])
        self.assertEqual(tanimoto(set1, set2), .0)
        # Similarity to set1 should be .5
        set3 = np.array([[1, 1, 1, 0, 0, 0]])
        self.assertEqual(tanimoto(set1, set3), .5)
        # Similarity to set1 should be .5
        set4 = np.array([[0, 0, 0, 1, 1, 1]])
        self.assertEqual(tanimoto(set1, set4), .5)
        # Similarity between set4 and set5 should be .5
        set5 = np.array([[0, 1, 0, 1, 0, 1]])
        self.assertEqual(tanimoto(set4, set5), .5)
        # set6 and set7 should be more similar than
        # set1 and set3, since 6 and 7 contain more 1s
        set6 = np.array([[1, 1, 1, 1, 1, 1, 1]])
        set7 = np.array([[1, 1, 1, 1, 1, 1, 0]])
        set8 = np.array([[1, 1, 1, 1, 1, 0]])
        self.assertTrue(tanimoto(set6, set7) > tanimoto(set1, set8))
        # However, set9 and set10 should have the same similarity as
        # set11 and set12 due to the same number of 1s
        set9 = np.array([[0, 0, 1, 1, 1, 1, 0, 0]])
        set10 = np.array([[0, 0, 1, 0, 1, 0, 0, 0]])
        set11 = np.array([[1, 1, 1, 1]])
        set12 = np.array([[1, 1, 0, 0]])
        self.assertTrue(tanimoto(set9, set10) == tanimoto(set11, set12))


if __name__ == '__main__':
    unittest.main()
