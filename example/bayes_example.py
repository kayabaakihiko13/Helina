from Hela.common.common import BayesTheorem
from typing import Type

# probability of an email spam (prior probability)
p_a: float = 0.2
# probability of observing certain keywords in a spam email
p_b_given_a: float = 0.9
# probability of observing certain keywords in a non spam
p_b_given_not_a: float = 0.1
#  create instance of the BayesTheorem
bayes_theorem: Type[BayesTheorem] = BayesTheorem()

# calculate the probability that an containing keyword spam
result: float = bayes_theorem.bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)

print(f"probability of spam given keyword is {result:.4f}")
