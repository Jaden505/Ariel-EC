# Standard libraries
from collections.abc import cast
from typing import TYPE_CHECKING
import numpy as np

# Local libraries
from ariel.ec.a001 import Individual
from ariel.operators.crossover import Crossover
from ariel.operators.mutation import FloatMutator

type Population = list[Individual]

def crossover(population: Population, RNG: np.random.Generator, lambda_: float, alpha: float) -> Population:
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


def mutation(population: Population, mutation_probability: float) -> Population:
    """Swap mutation"""
    for ind in population:
        if ind.tags.get("mut", False):
            genes = cast("list[float]", ind.genotype)
            mutated = FloatMutator.float_creep(
                individual=genes,
                span=1,
                mutation_probability=mutation_probability,
            )
            ind.genotype = mutated

    return population


def create_individual(RNG: np.random.Generator, controller_size: int) -> Individual:
    ind = Individual()
    
    genotype = {
        "morphology": [
            RNG.uniform(0, 1, size=64).astype(np.float32) for _ in range(3)
        ],
        "controller": RNG.normal(0, 0.5, size=(controller_size,)).astype(np.float32)
    }

    ind.genotype = genotype
    ind.tags = {"ps": False, "mut": False}
    ind.requires_eval = True
    return ind

