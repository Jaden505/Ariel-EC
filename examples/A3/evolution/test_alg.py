from evaluation import evaluate
from selection import parent_selection, survivor_selection
from variation import crossover, mutation, create_individual

from ariel.ec.a001 import Individual
from ariel.ec.a004 import EASettings, EAStep, EA

import numpy as np

type Population = list[Individual]
config = EASettings()

SEED = 42
RNG = np.random.default_rng(SEED)
MUTATION_PROBABILITY = 0.1
N_GENERATIONS = 50

def fitness_func(genotype: list[list[float]]) -> float:
    fitness = 0.0
    for vec, target_vec in zip(genotype, TARGET):
        fitness -= sum((g - t) ** 2 for g, t in zip(vec, target_vec))
    return fitness

def main() -> None:
    """Entry point."""
    # Create initial population
    population_list = [create_individual(RNG, VECTOR_LENGTH) for _ in range(10)]
    population_list = evaluate(population_list, fitness_func)

    # Create EA steps
    ops = [
        EAStep("parent_selection", parent_selection, {"mu": mu}),
        EAStep("crossover", crossover, {"RNG": RNG, "lambda_": lambda_, "alpha": alpha}),
        EAStep("mutation", mutation , {"mutation_probability": MUTATION_PROBABILITY}),
        EAStep("evaluation", evaluate, {"fitness_func": fitness_func}),
        EAStep("survivor_selection", survivor_selection, {"mu": mu}),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=100,
    )

    ea.run()
    
    best = ea.get_solution(only_alive=True)
    print("Best individual:", best)
    print("Best fitness:", best.fitness)
    print("Best genotype:", best.genotype)
    
# Test the mu lambda algorithm with n queens to ensure it works as expected
TARGET = [
    [0.2, 0.4, 0.6, 0.8],
    [0.1, 0.9, 0.3, 0.7],
    [0.5, 0.5, 0.5, 0.5]
]
VECTOR_LENGTH = 4

mu = 15
lambda_ = 100
alpha = 0.3

main()


