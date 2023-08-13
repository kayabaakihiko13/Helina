from Hela.constant import PI


class Math:
    """
    Curated list math
    """

    def factorial(self, n: int) -> int:
        """
        checking factorial of number

        Args:
            n (int): number of given

        Return:
            factorial of numbers

        Example:
        >>> Math().factorial(5)
        120
        """
        if not isinstance(n, int):
            raise ValueError("factorial(): only accept integral values")
        if n < 0:
            raise ValueError("factorial(): only accept integral values")
        return 1 if n in {0, 1} else n * self.factorial(n - 1)

    def radians(self, degree: float) -> float:
        """
        convert angle from degree to radians

        Args:
            degree (float): angle in degree

        Returns:
            (float): angle in radians

        Example:
        >>> Math().radians(90)
        1.5707963267948966
        """
        return degree / (180 / PI)

    def avg_mean(self, nums: list) -> float:
        """
        find mean of list of numbers

        Args:
            nums (list): list of numbers
        Returns:
            (float): mean of list of numbers

        Example:
        >>> Math().avg_mean([1, 2, 3, 4, 5])
        3.0
        """
        if not nums:
            raise ValueError("mean(): list empty")
        return sum(nums) / len(nums)

    def sine(
        self, angle_degree: float, accuracy: int = 18, rounded_value: int = 10
    ) -> float:
        """
        sine function

        Args:
            angle_degree (float): angle in degree
            accuracy (int, optional): number terms to calcylate the sine, defaults to 18
            rounded_value (int, optional): number of decimal places to round the result, default 10.

        Returns:
            (float): sine value of angle in degree

        Example:
        >>> sin(270.0)
        -1.0
        """
        angle_in_degree = angle_degree - ((angle_degree // 360.0) * 360.0)
        angle_in_radians = self.radians(angle_in_degree)
        result = angle_in_radians
        a = 3
        b = -1
        for _ in range(accuracy):
            result += (b * (angle_in_radians**a)) / self.factorial(a)
            b = -b
            a += 2

        return round(result, rounded_value)
