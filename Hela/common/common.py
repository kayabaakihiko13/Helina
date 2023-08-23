from __future__ import annotations
import numpy as np
from typing import Any, Union


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
        self._digits = [1]
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
        carry = 0
        for i in range(max(len(self._digits), len(other._digits))):
            if i < len(self._digits):
                digit_sum = self._digits[i] + other._digits[i] + carry
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
        minor = matrix[:row] + matrix[row + 1 :]
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
        det = Matrix.determinant(matrix)
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
        shape = Matrix._shape(matrix_a) + Matrix._shape(matrix_b)
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