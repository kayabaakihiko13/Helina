import math


class Differential(object):
    """
    utility class for calculating derivatives using various rules

    provides methods to calculate derivatives using the power rule, product rule
    quotien rule, and chain rule
    """

    @staticmethod
    def derivative(f, x: float, h: float = 1e-6) -> float:
        """
        calculate the derivative of a function using the limit definition

        Args:
            f (function): the function to calculate the derivateive of
            x (float): the point at which to calculate the derivative
            h (float, optional): the small value for the limit, default 1e-6

        Returns:
            (float): the calculated derivative

        Example
        >>> def f(x):
        ...     return x ** 2
        >>> diff = Differential()
        >>> result = diff.derivative(f, 2)
        >>> print(result)
        4.00000016666679
        """
        try:
            return (f(x + h) - f(x)) / h
        except Exception as error_derivative:
            raise ValueError(f"error {error_derivative}")

    @staticmethod
    def power_derivative(n: int, x: float) -> float:
        """
        calculate the derivative of x^n using the power rule

        Args:
            n (int): the exponent of the power function
            x (float): the value at which to calculate the derivative

        Returns:
            (float): the calculated derivative

        Example
        >>> diff = Differential()
        >>> result = diff.power_derivative(3, 2)
        >>> print(result)
        12
        """
        try:
            return n * math.pow(x, n - 1)
        except Exception as error_power_derivative:
            raise ValueError(f"error: {error_power_derivative}")

    @staticmethod
    def product_derivate(u, v, x: float) -> float:
        """
        calculate the derivative of the product of two functions u(x) and v(x)
        using product rule

        Args:
            u (function): first function u(x)
            v (function): second function v(x)
            x (float): the value at which to calculate the derivative

        Returns:
            (float): the calculated derivative

        Example
        >>> def f(x):
        ...     return x ** 2
        >>> def g(x):
        ...     return x ** 3
        >>> diff = Differential()
        >>> result = diff.product_derivate(f, g, 2)
        >>> print(result)
        64
        """
        try:
            u_derivative = u(x)
            v_derivative = v(x)
            return u_derivative * v(x) + u(x) * v_derivative
        except Exception as error_product_derivative:
            raise ValueError(f"error {error_product_derivative}")

    @staticmethod
    def quotient_derivate(u, v, x):
        """
        calculate the derivative of the quotient of two functions u(x) and v(x) using
        quotient rule

        Args:
            u (function): first numerator function u(x)
            v (function): second numerator function v(x)
            x (float): value at which to calculate the derivative

        Returns:
            (float): calculated derivative

        Example:
        >>> def f(x):
        ...     return x ** 2
        >>> def h(x):
        ...     return x
        >>> diff = Differential()
        >>> result = diff.quotient_derivate(f, h, 2)
        >>> print(result)
        1.0
        """
        try:
            u_derivative = u(x)
            v_derivative = v(x)
            return (v_derivative * u(x) - u_derivative * v(x)) / math.pow(v(x), 2)
        except Exception as error_quotient_derivative:
            raise ValueError(f"error: {error_quotient_derivative}")

    @staticmethod
    def composite_derivative(w, u, x: int) -> float:
        """
        calculate the derivative of the composite function w(u(x)) using chain rule

        Args:
            w (function): outer function w
            u (function): the input function u
            x (float): the value at which to calculate the derivative

        Returns:
            float: the calculated derivative

        Example
        >>> def g(x):
        ...     return x ** 3
        >>> def k(x):
        ...     return x ** 2 + 1
        >>> diff = Differential()
        >>> result = diff.composite_derivative(g, k, 2)
        >>> print(result)
        625
        """
        try:
            w_derivative = w(u(x))
            u_derivative = u(x)
            return w_derivative * u_derivative
        except Exception as error_composite_derivative:
            raise ValueError(f"error: {error_composite_derivative}")
