# TASK 9
# If we know the probability of a mutation occurring in a single individual, what will it be in a given population?
# Result: number of individuals with mutation in the population (not probability)
import numpy as np


def simulate_snp_frequency(n, p):
    return np.random.binomial(n, p)


# the contents of the block below will only be executed if the script is run directly,
# and not imported as a module --> = "__main__".

if __name__ == "__main__":
    # Simulation parameters
    # n - population number of people in the population
    # p - probability of occurrence of SNPs
    n = 3600  # int(input()) # number of individuals in the population: 3600
    p = 0.14  # float(input()) # probability of SNP occurrence: 0.14

    print("The simulated number of people in the population is:", simulate_snp_frequency(n, p))
# Each run of the program may give a different result due to the randomness of the simulation !!!

"""
NOTES:
np.random.binomial(n, p):
--Generates a single sample from a binomial distribution with parameters n and p.
The result is the number of successes (people who have the SNP) in a population of n people
with probability p per person.

In the above case:
A single experiment with 3600 samples, with a probability of success in a single sample of 14%.
"""