from __future__ import annotations
from Hela.common import differential, antidifferential
import numpy as np

if __name__ == "__main__":
    # calculate the general anti derivative
    print("calculate the gneeral antiderivative of a number wich:")
    print("- function to intergrate is lambda x: x ** 2")
    print("- lower limit of intergration is 3.0")
    print("- and upper limit of intergration is 2.0")
    print("- and the interval for calculation, (default is 4)")
    print("- use `trapezodial` for the method (optional: `riemann`)")
    print(
        f"the result are : {antidifferential.general_antiderivative(lambda x: x ** 2, 3.0, 4, method='trapezoidal')}\n"
    )

    # this for Defivative example
    print("Differential Derivative")

    def func(x: int) -> int:
        return x**2

    value_input: int = 3
    calculate_defivative: float = differential.derivative(func, value_input)
    print(f"result Derivative:{calculate_defivative:.3f}")

    # Diffetential Power Derivative
    print("\nPower Derivative")
    exponent_input: int = 6
    calculate_power_derivative = differential.power_derivative(exponent_input, 2)

    print(f"result Power Derivative:{calculate_power_derivative:.3f}")

    # Differential Product Derivative
    print("\nProduct Derivative")

    def a_func(x):
        return x**2

    def b_func(x):
        return x

    input_value: int = 3
    result = differential.product_derivate(a_func, b_func, 2)
    print(f"Result Product Derivate {result}")
    # Differential Derivative Quotient
    print("\nDerivative Quotient")

    def u_func(x):
        return x**2

    def v_func(x):
        return x

    result = differential.quotient_derivate(u_func, v_func, 2)
    print(f"Result of Derivative Quotient:{result}\n")

    # power rule antiderivative
    print("calculate the antiderivative of x ^ n using power rule")
    print("with exponent of the power function is 3")
    print("and the value to calculate the antiderivative is 2")
    print(f"result is : {antidifferential.PowerRule_antiderivative(3, 2)}")
