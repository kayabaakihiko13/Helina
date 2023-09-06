from Hela.common.differential import Differential

if __name__ == "__main__":

    # this for Defivative example
    print("Differential Derivative")
    func = lambda x : x ** 2
    value_input :int = 3
    calculate_defivative:float = Differential.derivative(func,value_input)
    print(f"result Derivative:{calculate_defivative:.3f}")
    
    # Diffetential Power Derivative
    print("\nPower Derivative")
    exponent_input: int = 6
    calculate_power_derivative = Differential.power_derivative(exponent_input,
                                                               2)

    print(f"result Power Derivative:{calculate_power_derivative:.3f}")

    # Differential Product Derivative
    print("\nProduct Derivative")
    a_func = lambda x: x ** 2
    b_func = lambda x: x
    input_value:int = 3
    result = Differential.product_derivate(a_func,
                                           b_func,2)
    print(f"Result Product Derivate {result}")
    # Differential Derivative Quotient
    print("\nDerivative Quotient")
    u_func:function = lambda x: x ** 2
    v_func:function = lambda x: x
    result = Differential.quotient_derivate(u_func,v_func,
                                            2)
    print(f"Result of Derivative Quotient:{result}")