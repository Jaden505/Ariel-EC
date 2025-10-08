# Standard libraries
import numpy as np

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EASettings, EAStep, EA

from evolution.evaluation import evaluate
from evolution.variation import crossover, mutation, create_individual
from evolution.selection import parent_selection, survivor_selection

from simulation import evolve_experiment

type Population = list[Individual]
config = EASettings()

SEED = 42
RNG = np.random.default_rng(SEED)
MUTATION_PROBABILITY = 0.1

lambda_ = 15 # Offspring_size
mu = 5 # Parent_size
alpha = 0.5 # BLX-alpha parameter balance exploration/exploitation higher = more exploration


# Configuration
config.is_maximisation = False  
config.target_population_size = lambda_
config.num_of_generations = 100
config.db_handling = "delete"

evolve_experiment(morphology, controller max_num_timesteps=300)

def solve_problem(quiet:bool=False) -> None:    
    # Create initial population
    population_list = [create_individual(mu) for _ in range(config.target_population_size)]
    population_list = evaluate(population_list)

    # Create EA steps
    ops = [
        EAStep("evaluation", evaluate),
        EAStep("parent_selection", parent_selection),
        EAStep("crossover", crossover),
        EAStep("mutation", mutation),
        EAStep("evaluation", evaluate),
        EAStep("survivor_selection", survivor_selection),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=config.num_of_generations,
        quiet=quiet,
    )

    return ea.get_solution(only_alive=True)

