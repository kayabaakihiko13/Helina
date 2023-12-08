import Hela.mathfunc as mathfunc

if __name__ == "__main__":
    print("evaluating polynomial at given value")
    print("with coeffcients of the polynomial is [2, 3, 1]")
    print(f"the result: {mathfunc._polyval([2, 3, 1], 2)}\n")

    print("evaluate error function using taylor series")
    print("with input value for the example is 1.0")
    print(f"the result: {mathfunc._erf_taylor(1.0)}\n")

    print("evaluate the complementary error function using polynomial approximation")
    print("with the input value are 1.0")
    print(f"the result is: {mathfunc._erfc_mid(1.0)}\n")

    print("evaluate the complementary error function using asympotic series")
    print("with the example input value are 1.0")
    print(f"the result is: {mathfunc._erfc_asymp(1.0)}\n")
