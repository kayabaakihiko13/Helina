import numpy as np
from Hela.math import GeometricMean, PowerIteration, Harmonic, FastFourierTransforms

from Hela.common.common import Gaussian
from Hela.common.distribution import HypergeometricDistribution

if __name__ == "__main__":
    # geometric mean example
    print("Geometric Mean")
    numbers1: list = [2, 4, 6, 8]
    decimal_num: list = [0.5, 1.5, 2.6, 3.5]

    result_geometric_mean = GeometricMean(numbers1)
    result_geometric_mean_decimal = GeometricMean(decimal_num)

    print("Result geometric mean:", result_geometric_mean)
    print("Result geometric mean (decimal):", result_geometric_mean_decimal)

    print("\nPower Iteration")
    # Power iteration example
    input_matrix: np.ndarray = np.array([[2, 1], [1, 3]])
    vector: np.ndarray = np.array([1, 0])
    # create an instance of the power iteration class
    power_iteration = PowerIteration(input_matrix, vector)
    eigenvalue, eigenvector = power_iteration()
    print("Input matrix:", input_matrix)
    print("Initial vector:", vector)
    print("Largest eigenvalue:", eigenvalue)
    print("Corresponding eigenvector", eigenvector)

    print("\nHarmonic Series")
    # Harmonic Series
    input_series: list = [2, 4, 4, 6, 2]
    # create an instance of the Harmonic class
    harmonic_mean_calc = Harmonic(input_series)
    # check if the input series froms an arithmetic series
    is_arithmetic = harmonic_mean_calc.is_series()
    msg = None
    if is_arithmetic:
        msg = "The input series forms an arithmetic series."
    else:
        msg = "The input series does not form an arithmetic series"

    harmonic_mean_value = harmonic_mean_calc.mean()
    print("Input Series:", input_series)
    print("Harmonic Mean", harmonic_mean_value)

    print("\nFast Fourier Transform")
    # fast fourier transform
    input_vector = np.array([-1, 2, 3, 0])
    fft_instance = FastFourierTransforms()
    # perform discrete FFT on the input vector
    output_vector = fft_instance.discrectefft(input_vector)
    print("Input vector:", input_vector)
    print("Output vector (Discrete FFT):", output_vector)

    x = np.arange(15)
    gauss = Gaussian(x)
    print(gauss.calculate_gaussian())
