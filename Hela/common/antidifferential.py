from __future__ import annotations
import numpy as np


class AntiDifferential(object):
    @staticmethod
    def general_antiderivative(
        f, a: int, b: int, num_interval: int = 4, method: str = "riemann"
    ) -> float:
        """
        ## Description

        general antiderivative is a one method of involve differentiation
        and general antiderivative is basic for learn or solving equation of
        involve differentation
        Args:
            f (functional): formula input of different formula or something else
            a (int): The lower limit of integration
            b (int): The upper limit of integration.
            num_interval (int, optional): Number of intervals for calculation.
                                          Defaults to 4.
            method (str): The method for computing the antiderivative. Options are "riemann" and "trapezoidal".
        Returns:
            result: The antiderivative result of the function.
        """
        if a >= b:
            raise ValueError("Your input a dont more than b value")
        try:
            delta_x = (b - a) / num_interval
            if method == "riemann":
                # setting result of antidifferential
                result = 0
                for i in range(num_interval):
                    x_i = a + i * delta_x
                    result += f(x_i) * delta_x
                return round(result)
            if method == "trapezoidal":
                # setting result of antidifferential
                result = 0.5 * (f(a) + f(b))
                for i in range(1, num_interval):
                    result += f(a + i * delta_x)
                return round(result * delta_x)

        except Exception as error_integral:
            raise ValueError(f"Error {error_integral}")
        return 0

    @staticmethod
    def PowerRule_antiderivative(n: int, x: int | float) -> float:
        """
        ## Description

        calculate the anti derivative of x^n using the power rule
        Args:
            n (int): the exponent of the power function
            x (int | float): the value at which to calculate the anti
                             derivative
        Returns:
            float: The antiderivative result of the Power Rule.
        """
        try:
            return np.power(x, n + 1) / (n + 1)
        except Exception as powerrule_antidiffential:
            raise ValueError(f"error :{powerrule_antidiffential}")
