class Euler:
    def totient(self, number: int) -> int:
        """
        calculate the totient function for a given number

        Args:
            number (int): the number whose totient function
                         is to be calculated

        Returns:
            (int): the totient function for the number
        """
        # initialize a list to store whether eac number is prime
        is_prime: list = [True for i in range(number + 1)]
        # initialize a list to store the totient of each number
        totients: list = [i - 1 for i in range(number + 1)]
        # initialize a list to store the primes
        primes: list = []

        # iterate over all nnumbers from 2 to number + 1
        for i in range(2, number + 1):
            if is_prime[i]:
                primes.append(i)

            # for each prime number is prime list, iterate over all members
            # that are multiple of the prime number
            for j in range(0, len(primes)):
                # if the current number is greater than or equal to the product
                # of the prime number and the current index, break out the loop
                if i * primes[j] >= number:
                    break
                # mark the curret number as not prime
                is_prime[i * primes[j]] = False

                # if the curretn number is divisible by the prime number, set
                # the totients of the current number of the product of the totient
                # of the prime number.
                if i % primes[j] == 0:
                    totients[i * primes[j]] = totients[i] * primes[j]
                    break

                # otherwise, set the totient of the current number of the product
                # of the totient of the prime number and the prime number minus 1
                totients[i * primes[j]] = totients[i] * (primes[j] - 1)

        # return the totient of the number
        return totients
