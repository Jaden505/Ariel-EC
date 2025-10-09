# Standard libraries
import numpy as np

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EASettings, EAStep, EA
from ariel.simulation.controllers.controller import Controller

from evolution.evaluation import evaluate
from evolution.variation import crossover, mutation, create_individual
from evolution.selection import parent_selection, survivor_selection

from simulation import evolve_simulation, nn_controller, NUM_OF_MODULES

type Population = list[Individual]
config = EASettings()

SEED = 42
RNG = np.random.default_rng(SEED)
MUTATION_PROBABILITY = 0.1
N_GENERATIONS = 10


lambda_ = 18 # Offspring_size
mu = 8 # Parent_size
alpha = 0.2 # BLX-alpha parameter balance exploration/exploitation higher = more exploration

# Configuration
config.is_maximisation = False  
config.target_population_size = lambda_
config.num_of_generations = N_GENERATIONS
config.db_handling = "delete"


def solve_problem(controller: Controller, evolve_morphology:bool=False, quiet:bool=False, 
                  best_individual: Individual = None) -> Individual:    
    # Create initial population
    population_list = [create_individual(RNG, NUM_OF_MODULES, best_individual) for _ in range(config.target_population_size)]
    population_list = evaluate(population_list, evolve_simulation, controller)

    # Create EA steps
    ops = [
        EAStep("evaluation", evaluate, {"fitness_func": evolve_simulation, "controller": controller}),
        EAStep("parent_selection", parent_selection, {"mu": mu}),
        EAStep("crossover", crossover, {"RNG": RNG, "lambda_": lambda_, "alpha": alpha}),
        EAStep("mutation", mutation, {"mutation_probability": MUTATION_PROBABILITY}),
        EAStep("evaluation", evaluate, {"fitness_func": evolve_simulation, "controller": controller}),
        EAStep("survivor_selection", survivor_selection, {"mu": mu}),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=config.num_of_generations,
        quiet=quiet,
    )
    ea.run()
    
    return ea.get_solution(only_alive=True)


if __name__ == "__main__":
    ctrl = Controller(nn_controller)
    for _ in range(N_GENERATIONS):  # Run 3 times: first without morphology evolution, then with morphology evolution twice
        best = solve_problem(ctrl, quiet=False)
    