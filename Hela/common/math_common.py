from __future__ import annotations
import numpy as np
import math
from typing import Any, Union, Iterable


class LargeNumber:
    """
    Class for handling large integer numbers with arbitrary precision
    """

    def __init__(self, number=0):
        """
        Default constructor that initializes the number.

        Args:
            number (int or float): The input integer or float to be converted to a LargeNumber.

        Example:
            >>> large_num = LargeNumber(12345)
            >>> large_num = LargeNumber(12.345)
        """
        self._digits: list = [1]
        if isinstance(number, int):
            for i in str(number).strip():
                self._digits.append(int(i))
        elif isinstance(number, float):
            for i in str(number).strip():
                self._digits.append(int(i))

    def __repr__(self):
        """
        Returns a string representation of the number.

        Returns:
            str: The string representation of the LargeNumber.

        Example:
            >>> large_num = LargeNumber(12345)
            >>> repr(large_num)
            '54321'
        """
        return "".join([str(i) for i in self._digits[::-1]])

    def __add__(self, other):
        """
        Returns a string representation of the number.

        Returns:
            str: The string representation of the LargeNumber.

        Example:
            >>> large_num = LargeNumber(12345)
            >>> repr(large_num)
            '54321'
        """
        if not isinstance(other, LargeNumber):
            raise ValueError("Can only add LargeNumber objects")

        result = LargeNumber()
        carry: int = 0
        for i in range(max(len(self._digits), len(other._digits))):
            if i < len(self._digits):
                digit_sum: int = self._digits[i] + other._digits[i] + carry
            else:
                digit_sum = other._digits[i] + carry
            carry = digit_sum // 10
            result._digits.append(digit_sum % 10)

        if carry:
            result._digits.append(carry)

        return result

    def __len__(self):
        """
        Adds two LargeNumbers.

        Args:
            other (LargeNumber): The other LargeNumber to be added.

        Returns:
            LargeNumber: The sum of the two LargeNumbers.

        Raises:
            ValueError: If the input is not a LargeNumber object.

        Example:
            >>> num1 = LargeNumber(12345)
            >>> num2 = LargeNumber(6789)
            >>> result = num1 + num2
            >>> repr(result)
            '54321'
        """
        return len(self._digits)


class Matrix:
    """
    class for matrix operations
    """

    @staticmethod
    def transpose(
        matrix: list[list[int]], return_map: bool = True
    ) -> list[list[int]] | map[list[int]]:
        """
        Transpose matrix

        Args:
            matrix (list[list[int]]): the input matrix to be transposed
            return_map (bool, optional): whether to return the transposed matrix as a map

        Return:
            (list[list[int]] or map[list[int]]): the transposed matrix

        Example
        >>> matrix = [[1, 2, 3], [4, 5, 6]]
        >>> Matrix.transpose(matrix)
        [[1, 4], [2, 5], [3, 6]]
        """
        if return_map:
            return map(list, zip(*matrix))
        else:
            return list(map(list, zip(*matrix)))

    @staticmethod
    def identity(n: int) -> list[list[int]]:
        """
        Create an identity matrix

        Args:
            n (int): the size of the identity matrix

        Return:
            (list[list[int]]): the identity matrix

        Example
        >>> Matrix.identity(3)
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        """
        n = int(n)
        return [[int(row == column) for column in range(n)] for row in range(n)]

    @staticmethod
    def scalar_multiply(matrix: list[list[int]], n: float) -> list[list[float]]:
        """
        multiply a matrix by a scalar

        Args:
            matrix (list[list[int]]): the input matrix to be multiplied
            n (float): the scalar value to multiply with

        Return:
            (list[list[float]]): the result of scalar multiplication

        Example
        >>> matrix = [[1, 2], [3, 4]]
        >>> Matrix.scalar_multiply(matrix, 2)
        [[2, 4], [6, 8]]
        """
        return [[x * n for x in row] for row in matrix]

    @staticmethod
    def minor(matrix: list[list[int]], row: int, column: int) -> list[list[int]]:
        """
        compute the minor matrix obtained by removing a specified row and column.
        Args:
            matrix (list[list[int]]): The input matrix.
            row (int): The row index to be removed.
            column (int): The column index to be removed.

        Returns:
            (list[list[int]]): The minor matrix after removing the specified row and column.

        Example
        >>> matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        >>> Matrix.minor(matrix, 1, 1)
        [[1, 3], [7, 9]]
        """
        minor: list = matrix[:row] + matrix[row + 1 :]
        return [row[:column] + row[column + 1 :] for row in minor]

    @staticmethod
    def determinant(matrix: list[list[int]]) -> Any:
        """
        compute the determinant of a square matrix

        Args:
            matrix (list[list[int]]): the input square matrix

        Return
            (Any): the determinant of the input matrix

        Example
        >>> matrix = [[2, 3], [4, 5]]
        >>> Matrix.determinant(matrix)
        -2
        """
        if len(matrix) == 1:
            return matrix[0][0]
        return sum(
            x * Matrix.determinant(Matrix.minor(matrix, 0, i)) * (-1) ** i
            for i, x in enumerate(matrix[0])
        )

    @staticmethod
    def inverse(matrix: list[list[int]]) -> list[list[float]] | None:
        """
        compute the inverse of a square matrix

        Args:
            matrix (list[list[int]]): the input square matrix

        Return:
            (list[list[float]] | None): inverse matrix if it exists, else None

        Example
        >>> matrix = [[2, 3], [4, 5]]
        >>> Matrix.inverse(matrix)
        [[-5.0, 3.0], [4.0, -2.0]]
        """
        det: float = Matrix.determinant(matrix)
        if det == 0:
            return None

        matrix_minor = [
            [Matrix.determinant(Matrix.minor(matrix, i, j)) for j in range(len(matrix))]
            for i in range(len(matrix))
        ]
        cofactors = [
            [x * (-1) ** (row + col) for col, x in enumerate(matrix_minor[row])]
            for row in range(len(matrix))
        ]
        adjugate = list(Matrix.transpose(cofactors))
        return Matrix.scalar_multiply(adjugate, 1 / det)

    @staticmethod
    def _shape(matrix: list[list[int]]) -> tuple[int, int]:
        """
        get shape of the matrix

        Args:
            matrix (list[list[int]]): the input matrix

        Return:
            (tuple[int, int]): the number of rows and columns of the matrix

        Example
        >>> matrix = [[1, 2, 3], [4, 5, 6]]
        >>> Matrix._shape(matrix)
        (2, 3)
        """
        return len(matrix), len(matrix[0])

    @staticmethod
    def _check_not_int(matrix: list[list[int]]) -> bool:
        """
        check if a matrix is not an integer matrix

        Args:
            matrix (list[list[int]]): the input matrix

        Return:
            (bool): true if the matrix is not an integer matrix, False otherwise
        """
        return not isinstance(matrix, int) and not isinstance(matrix[0], int)

    @staticmethod
    def _verify_matrix_size(
        matrix_a: list[list[int]], matrix_b: list[list[int]]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        verify the size compatibliy of two matrices for element-wise operations

        Args:
            matrix_a (list[list[int]]): the first input matrix
            matrix_b (list[list[int]]): the second input matrix

        Return
            (tuplep[tuple[int, int], tuple[int, int]]): tuple of shapes of the two matrices

        Example
        >>> matrix_a = [[1, 2], [3, 4]]
        >>> matrix_b = [[5, 6], [7, 8]]
        >>> Matrix._verify_matrix_size(matrix_a, matrix_b)
        ((2, 2), (2, 2))
        """
        shape: tuple[int, int, int, int] = Matrix._shape(matrix_a) + Matrix._shape(
            matrix_b
        )
        if shape[0] != shape[3] or shape[1] != shape[2]:
            msg = (
                "_verify_matrix_size(): operands could be broadcast together "
                f"({shape[0], shape[1]}), ({shape[2], shape[3]})"
            )
            raise ValueError(msg)
        return (shape[0], shape[2]), (shape[1], shape[3])


class Series:
    @staticmethod
    def hexagonal_num(length: int) -> list[int]:
        """
        return a list of hexagonal numbers up to the given length

        Args:
            length(int): the length of the list

        Returns:
            (list[int]): a list of hexagonal numbers

        Raises:
            ValueError: If `length` is not a positive integer.

        Example
        >>> Series.hexagonal_num(10)
        [1, 6, 15, 28, 45, 66, 91, 120, 153, 190]
        """
        if length <= 0 or not isinstance(length, int):
            raise ValueError("Common.hexagonal_num(): length must be positive integer")
        return [n * (2 * n - 1) for n in range(length)]

    @staticmethod
    def hexagonal(numbers: Union[list, tuple]) -> Union[list, tuple]:
        """
        Returns a list or tuple of hexagonal numbers form the given numbers

        Args:
            numbers(Union[list, tuple]): list of numbers
        Returns:
            (Union[list, tuple]): list or tuple of hexagonal numbers

        Raises:
            TypeError: If `numbers` is not a list or tuple.
            TypeError: If an element of `numbers` is not an integer.
            ValueError: If an element of `numbers` is not a positive integer.

        >>> input_list = [1, 2, 3, 4]
        >>> input_list = Series.hexagonal(input_list)
        [1, 6, 15, 28]
        """
        if not isinstance(numbers, (list, tuple)):
            raise TypeError(f"Series.hexagonal({numbers}) not list or tuple")

        hexagonal_numbers: list = []
        for number in numbers:
            if not isinstance(number, int):
                msg: str = f"input value of [number={number}] must integer"
                raise TypeError(msg)
            if number < 1:
                raise ValueError(f"number={number}, must be positive integer")
            hexagonal_numbers.append(number * (2 * number - 1))

        if isinstance(numbers, list):
            return hexagonal_numbers
        elif isinstance(numbers, tuple):
            return tuple(hexagonal_numbers)


class SignalAnalysis:
    @staticmethod
    def time_delay_embedding(
        time_series: np.ndarray, embedding_dimension: int, delay=1
    ) -> np.ndarray:
        """
        Perform time delay embedding on a time series.

        Time delay embedding is a technique used to reconstruct a higher-dimensional
        phase space from a one-dimensional time series. This is achieved by
        embedding the time series into an n-dimensional space,
        where n is the given order of the embedding. Each point in the
        reconstructed space is composed of the
        original value and its delayed copies.

        Args:
            x(np.ndarray): The input time series data as a one-dimensional numpy array
            order(int, optional): the order of the embedding, i.e, the dimension of
                                    the reconstructed space, default 3
            delay (int, optional): delay between consecutive values in the time series
                                    for embedding

        Returns:
            (np.ndarray): time-delay embedded data as a 2D numpy array, where each row
                            corresponds to a point in the reconstructed space

        Example
        >>> import numpy as np
        >>> time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> embedded_data = SignalAnalysis.time_delay_embedding(time_series, order=3, delay=2)
        >>> print(embedded_data)
        array([[1, 3, 5],
               [2, 4, 6],
               [3, 5, 7],
               [4, 6, 8],
               [5, 7, 9]])
        """
        if not isinstance(time_series, np.ndarray):
            raise TypeError("time_series must be a numpy.ndarray")
        if time_series.ndim != 1:
            raise ValueError("time_series must be one-dimensional")
        if embedding_dimension < 1:
            raise ValueError("embedding_dimension must be greater than or equal to 1")
        if delay < 1:
            raise ValueError("delay must be greater than or equal to 1")

        series_length: int = len(time_series)
        embedded_series: np.ndarray[float] = np.empty(
            (embedding_dimension, series_length - (embedding_dimension - 1) * delay)
        )
        for i in range(embedding_dimension):
            embedded_series[i] = time_series[
                i * delay : i * delay + embedded_series.shape[1]
            ]
        return embedded_series.T


class Sigmoid:
    def __init__(self, vector: list):
        """
        Initialize the sigmoid function calculator

        This class computes the sigmoid values for a given input vector
        of numerical values. the class takes vector of real numbers as input
        and then 1 / (1 + exp(-x)). after through sigmoid, the element of the
        vector mostly 0 between 1. or 1 between -1.

        Args:
            vector (list): A list of numerical values for sigmoid calculation

        Example
        >>> sigmoid = Sigmoid([0.5, -1.0, 2.0])
        """
        self.vector = vector

    def calculate_sigmoid(self) -> list:
        """
        Calculation sigmoid values for the input vector

        Returns:
            (list): A list sigmoid values corresponds to the input vector

        Example
        >>> sigmoid = Sigmoid([0.5, -1.0, 2.0])
        >>> print(sigmoid.calculate_sigmoid())
        [0.6224593312018546, 0.2689414213699951, 0.8807970779778823]
        """
        return [1 / (1 + math.exp(-x)) for x in self.vector]

    def __repr__(self) -> str:
        """
        Return a string representation of the sigmoid instance

        Returns:
            (str): a string representation of the sigmoid function

        Example
        >>> sigmoid = Sigmoid([0.5, -1.0, 2.0])
        >>> print(sigmoid)
        Sigmoid([0.5, -1.0, 2.0])
        """
        return f"Sigmoid({self.vector})"


class Gaussian:
    def __init__(self, x, mu: float = 0.0, sigma: float = 1.0):
        """
        initialize a guassian

        this class provides methods for calculating the value of the
        gaussian function at a specified input, based on given mean and
        standard deviation

        Args:
            x (float): the input value at which to calculate the gaussian function
            mu (float): the mean (average) of the gaussian distribution
            sigma (float): the standard deviation of the gaussian distribution
        """
        self.x: float = x
        self.mu: float = mu
        self.sigma: float = sigma

    def calculate_gaussian(self) -> list[float]:
        """
        calculate the value of the gaussian function at the specified input

        Return:
            float: the value of the gaussian function at the given input

        Example
        >>> gaussian_calc = Gaussian(x=2.0, mu=0.0, sigma=1.0)
        >>> gaussian_value = gaussian_calc.calculate_gaussian()
        >>> print(f"Gaussian value at x = {gaussian_calc.x}: {gaussian_value:.6f}")
        Gaussian value at x = 2.0: 0.053990
        """
        result: list[float] = (
            1
            / np.sqrt(2 * np.pi * self.sigma**2)
            * np.exp(-((self.x - self.mu) ** 2) / (2**self.sigma**2))
        )
        return result

    def __repr__(self) -> str:
        """
        return a string representation of the gaussian instance

        Return:
            (str): a string representation of the gaussian instance
        """
        return f"Gaussian(x={self.x}, mu={self.mu}, sigma={self.sigma})"


class ReLU:
    def __init__(self, vector: list[int | float]) -> None:
        """
        # Description

        Relu is a function receives any negative input,
        it returns 0,the function any positives value x,it
        returns that value.As result,the output has a range of 0
        to infinite
        Args:
            vector (list[int | float]) : input values
        """
        self.vec: list[int | float] = vector

    def calculate_ReLu(self) -> list[int | float]:
        """
        # Description
        calculate Relu for the input Vector
        Returns:
            list[float]: A list of Relu at given input

        # example
        >>> from Hela.common.common import ReLU
        >>> a = [1,2,3,4]
        >>> ReLU(a).calculate_ReLu()
        [1, 2, 3, 4]
        >>> b = [-1,2,3,4]
        >>> ReLU(b).calculate_ReLu()
        [0, 2, 3, 4]
        >>> # for know form origional b is
        >>> ReLU(b)
        RelU([-1, 2, 3, 4])
        """
        return [max(0, i) for i in self.vec]

    def __repr__(self) -> str:
        return f"RelU({self.vec})"


class Logistic_map:
    def __init__(self, n: int, learning_path: float) -> None:
        """
        # Description
        Logistic Map is `Polynominal mapping`,the usual values
        of interest for the parameter r are those in the interval [0,4],
        so that `Xn` remains bounded on [0,1].
        Args:
        n (int) :input value for equivalently
        learning_path (float): step in itterable

        Example:
        >>> from Hela.common.common import Logistic_map
        >>> Logistic_map(10,0.001).calculate()
        -0.09
        """
        self.n: int = n
        self.learning_path: float = learning_path

    def calculate(self) -> float:
        """
        calculate Logistic Map for input vector
        Returns:
            float: the value of calculate for logistic Map
        """
        return self.learning_path * self.n * (1 - self.n)

    def show_iterable(self) -> str:
        result_str: str = ""
        for i in range(1, self.n + 1):
            result_str += f"iter: {i} | result = {self.calculate()}\n"
        return result_str


class BayesTheorem:
    @staticmethod
    def bayes_theorem(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
        """
        calculate the probabilities using bayes theorem

        baye's theorem calculates the probability of an event a happening given that
        event B has occured

        Args:
            p_a (float): probability of event A
            p_b_given_a (float): the probability of event B given that event A has occured
            p_b_given_not_a (float): the probability of event B given that event A has not occured

        Returns:
            (float): the probability of event A given B (P(A|B))

        Example:
        >>> BayesTheorem.bayes_theorem(0.3, 0.7, 0.4)
        0.4286
        """
        if (
            not (0 <= p_a <= 1)
            or not (0 <= p_b_given_a <= 1)
            or not (0 <= p_b_given_not_a <= 1)
        ):
            raise ValueError("probabilities must be in the range [0, 1]")

        # calculate P(B)
        p_b: float = (p_a * p_b_given_a) + ((1 - p_a) * p_b_given_not_a)
        # calculate P(A|B) using bayes theorem
        p_a_given_b: float = (p_a * p_b_given_a) / p_b if p_b != 0 else 0
        return p_a_given_b

class GeometricMean:
    """
    Calculate the geometric mean of a list of numbers.

    The geometric mean of a set of numbers is calculated by
    multiplying all the numbers in the input listand then
    taking the n-th root of the product, where n is
    the number of elements in the list.

    Args:
        series (list or tuple): List of numbers for which to
                                calculate the geometric mean.

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

    def __init__(self, series: list) -> None:
        self.series = series

    def __call__(self) -> float:
        """
        Calculate the geometric mean of a list of numbers.

        The geometric mean of a set of numbers is calculated by
        multiplying all the numbers in the input list and
        then taking the n-th root of the product,
        where n is the number of elements in the list.

        Args:
            series (list or tuple): List of numbers for which
                                    to calculate the geometric mean.

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
        if not isinstance(self.series, (list, tuple)):
            raise ValueError(
                "GeometricMean(): input series not valid - valid series - [2, 4, 8]"
            )
        if len(self.series) == 0:
            raise ValueError("GeometricMean(): input list must be a non empty list")
        answer = 1
        for value in self.series:
            answer *= value
        return math.pow(answer, 1 / len(self.series))

    def __repr__(self):
        return f"{self.__call__()}"


class PowerIteration:
    """
    Calculate the largest eigenvalue and corresponding eigenvector of a given matrix using the Power Iteration method.

    This class implements the Power Iteration algorithm
    to find the largest eigenvalue and its corresponding eigenvector
    of a given square matrix. The algorithm iteratively applies matrix-vector
    multiplications and normalization to converge
    towards the dominant eigenvalue and eigenvector.

    Args:
        error_total (float, optional): The desired tolerance for convergence.
                                       Defaults to 1e-12.
        max_iteration (int, optional): The maximum number of iterations before termination.
                                       Defaults to 100.
    Methods:
        __call__(self, input_matrix: np.ndarray,
                 vector: np.ndarray) -> tuple[float, np.ndarray]:
            Calculate the largest eigenvalue and corresponding eigenvector
            using Power Iteration.

    Example:
    >>> input_matrix = np.array([[2, 1], [1, 3]])
    >>> vector = np.array([1, 0])
    >>> power_iteration = PowerIteration()
    >>> eigenvalue, eigenvector = power_iteration(input_matrix, vector)
    """

    def __init__(
        self,
        input_matrix: np.ndarray,
        vector: np.ndarray,
        error_total: float = 1e-12,
        max_iteration: int = 100,
    ):
        self.input_matrix = input_matrix
        self.vector = vector
        self.error_total = error_total
        self.max_iteration = max_iteration

    def __call__(
        self,
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
        assert np.shape(self.input_matrix)[0] == np.shape(self.input_matrix)[1]
        assert np.shape(self.input_matrix)[0] == np.shape(self.vector)[0]
        assert np.iscomplexobj(self.input_matrix) == np.iscomplexobj(self.vector)
        is_complex = np.iscomplexobj(self.input_matrix)
        if is_complex:
            assert np.array_equal(self.input_matrix, self.input_matrix.conj().T)
        convergence = False
        lambda_previous = 0
        iterations = 0
        error = 1e12

        while not convergence:
            w = np.dot(self.input_matrix, self.vector)
            vector = w / np.linalg.norm(w)
            vector_h = vector.conj().T if is_complex else vector.T
            lambda_ = np.dot(vector_h, np.dot(self.input_matrix, self.vector))
            error = np.abs(lambda_ - lambda_previous) / lambda_
            iterations += 1

            if error <= self.error_total or iterations >= self.max_iteration:
                convergence = True
            lambda_previous = lambda_

        if is_complex:
            lambda_ = np.real(lambda_)
        return lambda_, vector

    def __repr__(self):
        return f"{self.__call__()}"


class Harmonic:
    """
    calculating the harmonic mean of a series numbers

    Example:
    >>> series = [2, 4, 6]
    >>> harmonic_mean = Harmonic(seris)
    >>> harmonic_mean()
    3.4285714285714284
    >>> harmonic_mean.series()
    """

    def __init__(self, series: list) -> None:
        """
        initialize the harmonic object with the input series

        Args:
            series(list): the list of numbers for which to calculate the
                         harmonic mean
        """
        self.series = series

    def is_series(self) -> bool:
        """
        check if the input series forms an arithmetic series

        Returns:
            (bool): true if the input series froms an arithmetic series, False otherwise

        Raises:
            ValueError: if the input series is not valid or doesn't an arithmetic series
        """
        if not isinstance(self.series, (list, tuple)):
            raise ValueError(
                f"HarmonicMean({self.series}).series() not valid, valid series - [1, 2/3, 2]"
            )
        if len(self.series) == 0:
            raise ValueError(
                f"HarmonicMean({self.series}).series() input list tuple must be an non empty"
            )
        if len(self.series) == 1 and self.series[0] != 0:
            return True

        rec_series: list = []
        series_len: int = len(self.series)
        for i in range(0, series_len):
            if self.series[i] == 0:
                raise ValueError(
                    f"HarmonicMean.series({self.series[0]}) cannot have 0 as an element"
                )
            rec_series.append(1 / self.series[i])
        common_dif = rec_series[1] - rec_series[0]
        for index in range(2, series_len):
            if rec_series[index] - rec_series[index - 1] != common_dif:
                return False
        return True

    def mean(self) -> float:
        """
        calculate the harmonic mean of the input series

        Returns:
            float: the harmonic mean of the input series

        Raises:
            ValueError: if the input series is not valid or empty
        """
        if not isinstance(self.series, (list, tuple)):
            raise ValueError(
                f"HarmonicMean({self.series}).mean(): input series not valid, valid series - [2, 4, 6]"
            )
        if len(self.series) == 0:
            raise ValueError(
                f"HarmonicMean({self.series}).mean(): input (list, tuple) must be a non empty"
            )
        answer: int = 0
        for val in self.series:
            answer += 1 / val
        return len(self.series) / answer


class Signal:
    """
    Calculate the singal processing
    signal processing is an electrical engineer subfield that
    focuses on analyzing.
    """

    def __init__(self, series: np.ndarray, control_matrix: np.array) -> None:
        self.series = series
        self.control_matrix = control_matrix

    def cspline1d(self, idx: int, x: int, p: int) -> float:
        """
        CsSpline1D is a Polynominal-time and Numerically stable
        algorithm for evaluating spline curves
        Args:
            control_matrix (np.array): index of knot interval that
                                       contains x.
            idx (int): Index of knot interval that contains x.
            x (int): Position.
            p (int): Degree of B-spline.

        Returns:
            float: value of evaluating spline curve
        """
        d = np.zeros(p + 1)
        for r in range(0, p + 1):
            r_adjusted = r + idx - p  # Adjusted index
            if 0 <= r_adjusted < len(self.control_matrix):  # Check for valid index
                d[r] = self.control_matrix[r_adjusted]

        for r in range(1, p + 1):
            for j in range(p, r - 1, -1):
                alpha = x - self.series[j + idx]
                denominator = self.series[j + 1 + idx - r]
                -self.series[j + idx - p]
                if denominator != 0:
                    alpha /= denominator
                    d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]

        return d[p]

    def cspline2d(self, eval_point: float, idx: int, degree: int) -> float:
        """
        Cox-de Boor is recursion formula to support in B-splines with
        Polynominals are positive in a finite domain and zero elsewhere

        Args:
            eval_point (float): Evaluation point
            idx (int): Index of knot interval
            degree (int): Degree of B-spline.

        Returns:
            float: Value of evaluting spline curve
        """
        if degree == 0:
            if (self.series[idx] <= eval_point) and (eval_point < self.series[idx + 1]):
                return 1
            return 0

        alpha = (eval_point - self.series[idx]) / (
            self.series[idx + degree] - self.series[idx]
        )

        left = alpha * self.cspline2d(eval_point, idx + 1, degree - 1)
        right = (1 - alpha) * self.cspline2d(eval_point, idx + 1, degree - 1)

        return float(left + right)


class FastFourierTransforms:
    """Fast Fourier Transforms is algorithm that computes
    `the Discrate Fourier Transform`
    """

    def discrectefft(self, vector: np.ndarray) -> np.ndarray:
        """
        Discrecte FFT is a finite of equally-spaced samples
        of a function into a same-length sequence of equaly-
        spaced sample of Fast Fouruier Transforms
        """
        n = len(vector)
        if n <= 1:
            return vector
        x_even = self.discrectefft(vector[0::2])
        x_odd = self.discrectefft(vector[1::2])
        factor = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate(
            [
                x_even + factor[: n // 2] * x_odd,
                x_even + factor[n // 2 :] * x_odd,
            ]
        )
