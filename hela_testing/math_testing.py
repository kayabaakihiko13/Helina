import unittest
import numpy as np
from Hela.math import GeometricMean, PowerIteration, Harmonic


class TestGeometricMean(unittest.TestCase):
    def test_geometric_mean(self):
        gm = GeometricMean([2, 4, 8])
        result = gm()

        expected_result = 4.0
        self.assertAlmostEqual(result, expected_result)

    def test_geometric_mean_float(self):
        gm = GeometricMean([4, 8, 16])
        result = gm()
        expected_result = 7.999999999999999
        self.assertEqual(result, expected_result)

    def test_geometric_mean_zero(self):
        gm = GeometricMean([0, 2, 3])
        result = gm()
        expected_result = 0.0
        self.assertEqual(result, expected_result)


class TestPowerIteration(unittest.TestCase):
    def test_power_iteration_real_matrix(self):
        input_matrix = np.array([[2, 1], [1, 3]])
        vector = np.array([1, 0])
        power_iteration = PowerIteration(input_matrix, vector)
        eigenvalue, eigenvector = power_iteration()

        expected_eigenvalue = 2.23606797749979
        expected_eigenvector = np.array([0.37139068, 0.92847669])

        self.assertAlmostEqual(eigenvalue, expected_eigenvalue, places=12)


class TestHarmonic(unittest.TestCase):
    def test_is_series_valid(self):
        valid_series = [2, 4, 6]
        harmonic_mean = Harmonic(valid_series)
        self.assertFalse(harmonic_mean.is_series())

    def test_mean_valid(self):
        valid_series = [1, 4, 4]
        harmonic_mean = Harmonic(valid_series)
        self.assertEqual(harmonic_mean.mean(), 2.0)

    def other_test_mean_valid(self):
        valid_series = [3, 6, 9, 12]
        harmonic_mean = Harmonic(valid_series)
        self.assertEqual(harmonic_mean.mean(), 5.759999999999999)

    def test_mean_valid_dummy(self):
        valid_series = [1, 2, 3]
        harmonic_mean = Harmonic(valid_series)
        self.assertEqual(harmonic_mean.mean(), 1.6363636363636365)
