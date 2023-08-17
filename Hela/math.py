import math
import numpy as np


class GeometricMean:
    """
    Calculate the largest eigenvalue and corresponding eigenvector of matrix `input_matrix`
    given a random vector in the same space.

    Args:
        input_matrix (np.ndarray): The square matrix for which to calculate the largest eigenvalue and eigenvector.
        vector (np.ndarray): The initial vector used in the Power Iteration algorithm.

    Returns:
        tuple[float, np.ndarray]: The calculated largest eigenvalue and the corresponding eigenvector.

    Raises:
        AssertionError: If the dimensions of the input matrix and vector do not match.
        ValueError: If the matrix or vector is complex and not Hermitian (not equal to its conjugate transpose).

    Example:
        >>> input_matrix = np.array([[2, 1], [1, 3]])
        >>> vector = np.array([1, 0])
        >>> power_iteration = PowerIteration()
        >>> eigenvalue, eigenvector = power_iteration(input_matrix, vector)
    """

    def __init__(self):
        pass

    def __call__(self, series: list) -> float:
        """
        Calculate the geometric mean of a list of numbers.

        The geometric mean of a set of numbers is calculated by multiplying all the numbers in the input list
        and then taking the n-th root of the product, where n is the number of elements in the list.

        Args:
            series (list or tuple): List of numbers for which to calculate the geometric mean.

        Returns:
            float: Geometric mean of the input list.

        Example:
            >>> geometric_mean = GeometricMean()
            >>> geometric_mean([2, 4, 8])
            4.0
            >>> geometric_mean([1, 2, 3, 4, 5])
            2.605171084697352
            >>> geometric_mean([10, 20, 30, 40, 50])
            27.866004174074643

        Raises:
            ValueError: If the input series is not a valid list or tuple.
            ValueError: If the input list is empty.
        """
        if not isinstance(series, (list, tuple)):
            raise ValueError(
                "GeometricMean(): input series not valid - valid series - [2, 4, 8]"
            )
        if len(series) == 0:
            raise ValueError("GeometricMean(): input list must be a non empty list")
        answer = 1
        for value in series:
            answer *= value
        return math.pow(answer, 1 / len(series))


class PowerIteration:
    """
    Calculate the largest eigenvalue and corresponding eigenvector of a given matrix using the Power Iteration method.

    This class implements the Power Iteration algorithm to find the largest eigenvalue and its corresponding eigenvector
    of a given square matrix. The algorithm iteratively applies matrix-vector multiplications and normalization to converge
    towards the dominant eigenvalue and eigenvector.

    Args:
        error_total (float, optional): The desired tolerance for convergence. Defaults to 1e-12.
        max_iteration (int, optional): The maximum number of iterations before termination. Defaults to 100.

    Methods:
        __call__(self, input_matrix: np.ndarray, vector: np.ndarray) -> tuple[float, np.ndarray]:
            Calculate the largest eigenvalue and corresponding eigenvector using Power Iteration.

    Example:
        >>> input_matrix = np.array([[2, 1], [1, 3]])
        >>> vector = np.array([1, 0])
        >>> power_iteration = PowerIteration()
        >>> eigenvalue, eigenvector = power_iteration(input_matrix, vector)
    """

    def __init__(self, error_total: float = 1e-12, max_iteration: int = 100):
        self.error_total = error_total
        self.max_iteration = max_iteration

    def __call__(
        self, input_matrix: np.ndarray, vector: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """
        Calculate the largest eigenvalue and corresponding eigenvector of matrix `input_matrix`
        given a random vector in the same space.

        Args:
            input_matrix (np.ndarray): The square matrix for which to calculate the largest eigenvalue and eigenvector.
            vector (np.ndarray): The initial vector used in the Power Iteration algorithm.

        Returns:
            tuple[float, np.ndarray]: The calculated largest eigenvalue and the corresponding eigenvector.

        Raises:
            AssertionError: If the dimensions of the input matrix and vector do not match.
            ValueError: If the matrix or vector is complex and not Hermitian (not equal to its conjugate transpose).

        Example:
            >>> input_matrix = np.array([[2, 1], [1, 3]])
            >>> vector = np.array([1, 0])
            >>> power_iteration = PowerIteration()
            >>> eigenvalue, eigenvector = power_iteration(input_matrix, vector)
        """
        assert np.shape(input_matrix)[0] == np.shape(input_matrix)[1]
        assert np.shape(input_matrix)[0] == np.shape(vector)[0]
        assert np.iscomplexobj(input_matrix) == np.iscomplexobj(vector)
        is_complex = np.iscomplexobj(input_matrix)
        if is_complex:
            assert np.array_equal(input_matrix, input_matrix.conj().T)
        convergence = False
        lambda_previous = 0
        iterations = 0
        error = 1e12

        while not convergence:
            w = np.dot(input_matrix, vector)
            vector = w / np.linalg.norm(w)
            vector_h = vector.conj().T if is_complex else vector.T
            lambda_ = np.dot(vector_h, np.dot(input_matrix, vector))
            error = np.abs(lambda_ - lambda_previous) / lambda_
            iterations += 1

            if error <= self.error_total or iterations >= self.max_iteration:
                convergence = True
            lambda_previous = lambda_

        if is_complex:
            lambda_ = np.real(lambda_)
        return lambda_, vector
