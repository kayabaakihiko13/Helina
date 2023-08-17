import numpy as np
from Hela.math import PowerIteration, GeometricMean


def test_power_iteration() -> None:
    input_matrix = np.array([[41, 4, 20], [4, 26, 30], [20, 30, 50]])
    vector = np.array([41, 4, 20])
    test = PowerIteration()

    print(test.__call__(input_matrix, vector))


def test_geometric_mean() -> None:
    test = GeometricMean().__call__([2, 4, 8])
    print(test)


test_geometric_mean()
test_power_iteration()
