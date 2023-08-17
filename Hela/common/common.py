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
