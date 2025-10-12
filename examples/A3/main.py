# Standard libraries
import numpy as np

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EASettings, EAStep, EA

from evolution.evaluation import evaluate
from evolution.variation import crossover_controller, mutation_controller, create_individual, mutation_body, crossover_body
from evolution.selection import parent_selection, survivor_selection

from simulation import NUM_OF_MODULES
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

type Population = list[Individual]
config = EASettings()

SEED = 42
RNG = np.random.default_rng(SEED)
MUTATION_PROBABILITY = 0.1
N_GENERATIONS = 10
N_ITERATIONS = 10

BODY_VECTOR_SIZE = 64 
CONTROLLER_VECTOR_SIZE = NUM_OF_MODULES

lambda_ = 18 # Offspring_size
mu = 5 # Parent_size
alpha = 0.5 # BLX-alpha

# Configuration
config.is_maximisation = False  
config.target_population_size = lambda_
config.num_of_generations = N_GENERATIONS
config.db_handling = "delete"

def run_controller_evolution(nde: NeuralDevelopmentalEncoding, population_list: list[Population], quiet:bool=False):    
    """ Run evolution to optimize the controller nn weights while keeping the body fixed. """
    # Create EA steps
    ops = [
        EAStep("evaluation", evaluate, {"nde": nde}),
        EAStep("parent_selection", parent_selection, {"mu": mu}),
        EAStep("crossover", crossover_controller, {"RNG": RNG, "lambda_": lambda_, "alpha": alpha}),
        EAStep("mutation", mutation_controller, {"mutation_probability": MUTATION_PROBABILITY}),
        EAStep("evaluation", evaluate, {"nde": nde}),
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
    
    best = ea.get_solution(only_alive=True)
    return best.genotype['controller'], best.genotype['body']

def run_body_variation(best_controller: np.ndarray, population_list: Population) -> Population:
    """ Create new population by varying the body of the individuals while keeping the controller fixed to the best one found so far. """
    population_list = [create_individual(RNG, BODY_VECTOR_SIZE, CONTROLLER_VECTOR_SIZE) for _ in range(mu)]
    population_list = mutation_body(population_list, MUTATION_PROBABILITY)
    population_list = crossover_body(population_list, RNG, lambda_)
    
    for i in population_list:
        i.genotype['controller'] = best_controller
        
    return population_list


if __name__ == "__main__":
    population_list = [create_individual(RNG, BODY_VECTOR_SIZE, CONTROLLER_VECTOR_SIZE) for _ in range(lambda_)]
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    
    for i in range(N_ITERATIONS):
        print(f"--- Iteration {i+1}/{N_ITERATIONS} ---")
        best_controller, best_body = run_controller_evolution(nde)
        population_list = run_body_variation(best_controller, population_list)
            
    