from __future__ import annotations
from Hela.common import differential

if __name__ == "__main__":
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
    print(f"Result of Derivative Quotient:{result}")
