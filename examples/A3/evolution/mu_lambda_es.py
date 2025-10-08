# Standard libraries
import random
from collections.abc import cast, Callable
import numpy as np

# Local libraries
from ariel.ec.a000 import FloatMutator
from ariel.ec.a001 import Individual
from ariel.ec.a005 import Crossover
from ariel.ec.a004 import EASettings, EAStep, EA

type Population = list[Individual]
config = EASettings()

SEED = 42
RNG = np.random.default_rng(SEED)

lambda_ = 15 # Offspring_size
mu = 5 # Parent_size
alpha = 0.5 # BLX-alpha parameter

MUTATION_PROBABILITY = 0.1

def parent_selection(population: Population) -> Population:
    """Uniform selection"""
    random.shuffle(population)
    selected = population[:mu]
    
    for ind in selected:
        ind.tags["ps"] = True  # Mark as selected for parenthood
    
    return selected


def crossover(population: Population) -> Population:
    """BLX-alpha crossover creates lambda_ offspring"""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    offspring = []
    
    while len(offspring) < lambda_:
        parent1, parent2 = cast("Individual", RNG.choice(parents, size=2, replace=False))
        child1_genotype, child2_genotype = Crossover.blx_alpha(
            parent1.genotype,
            parent2.genotype,
            alpha=alpha,
        )
        
        child1 = Individual()
        child1.genotype = child1_genotype
        child1.tags = {"ps": False, "mut": True}  # Mark as needing mutation
        child1.requires_eval = True
        
        child2 = Individual()
        child2.genotype = child2_genotype
        child2.tags = {"ps": False, "mut": True}  # Mark as needing mutation
        child2.requires_eval = True
        
        offspring.extend([child1, child2])
    
    return offspring[:lambda_]


def mutation(population: Population) -> Population:
    """Swap mutation"""
    for ind in population:
        if ind.tags.get("mut", False):
            genes = cast("list[float]", ind.genotype)
            mutated = FloatMutator.float_creep(
                individual=genes,
                span=1,
                mutation_probability=MUTATION_PROBABILITY,
            )
            ind.genotype = mutated

    return population


def evaluate(population: Population, fitness_func: Callable[[list[float]], float]) -> Population:
    """Evaluate individuals that require evaluation"""
    for ind in population:
        if ind.requires_eval:
            ind.fitness = fitness_func(ind.genotype) 
            ind.requires_eval = False
    return population


def survivor_selection(population: Population) -> Population:
    """Elitism"""
    non_parents = [ind for ind in population if not ind.tags.get("ps", False)]
    sorted_population = sorted(non_parents, key=lambda ind: ind.fitness, reverse=True)
    
    # Select the top mu individuals
    selected = sorted_population[:mu]
    
    # Reset tags for the next generation
    for ind in selected:
        ind.tags = {"ps": False, "mut": False}
    
    return selected


def create_individual(n: int) -> Individual:
    ind = Individual()
    ind.genotype = RNG.uniform(
            low=0,
            high=1,
            size=n,
        )
    ind.tags = {"ps": False, "mut": False} # Initially not selected for anything
    return ind


def learning(population: Population) -> Population:
    return population

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
        EAStep("learning", learning),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=config.num_of_generations,
        quiet=quiet,
    )

    return ea.get_solution(only_alive=True)

# Configuration
config.is_maximisation = False  
config.target_population_size = lambda_
config.num_of_generations = 100
config.db_handling = "delete"

