import cmath
from typing import Union
from Hela import mathfunc

if __name__ == "__main__":

    def square(x: Union[float, int]) -> Union[float, int]:
        return x**2

    def square_complex(x: complex) -> complex:
        return x**2 + 1j

    # create a math function using square function
    math_square = mathfunc._mathfunction_real(square, square_complex)
    math_complex_function = mathfunc._mathfunction_real(square, square_complex)
    # test the math function with real number
    result_math_square_function = math_square(3.5)
    result_math_complex_function = math_complex_function(3.0 + 2.0j)

    print(f"result of math square function {result_math_square_function}")
    print(f"result math complex function {result_math_complex_function}")

    # greatest common divisor
    gcd = mathfunc.gcd(1, 7)
    print(f"result of the greatest common divisor of 1 and 7 is: {gcd}")
    # gcd_result = mathfunc.gcd(a=12, b=13)
    # print(gcd_result)

    # modular division
    mod_div = mathfunc.mod_division(4, 11, 5)
    print(f"result of modular division 4, 11, 5 is: {mod_div}")

    # modular division with real number
    mod_div = mathfunc.mod_division(4, 11.2, 5.2, precision=2)
    print(f"result of modular division 4, 11, 5 is: {mod_div}")

    # lcm
    lcm_result = mathfunc.lcm(12, 20)
    print(f"result of lcm is : {lcm_result}")
