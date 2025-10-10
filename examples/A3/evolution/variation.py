from typing import cast
import numpy as np
from ariel.ec.a001 import Individual
from ariel.ec.a000 import FloatMutator
from ariel.ec.a005 import Crossover

type Population = list[Individual]

# --- CREATE ---
def create_individual(RNG: np.random.Generator, vector_length: int) -> Individual:
    genotype = [
        RNG.uniform(0, 1, size=vector_length).astype(np.float32).tolist()
        for _ in range(3)
    ]
    ind = Individual()
    ind.genotype = genotype  # list[list[float]]
    ind.tags = {"ps": False, "mut": False}
    ind.requires_eval = True
    return ind

# --- CROSSOVER ---
def crossover(population: Population,
              RNG: np.random.Generator,
              lambda_: int,
              alpha: float) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    offspring = []

    while len(offspring) < lambda_:
        p1, p2 = RNG.choice(parents, size=2, replace=False)
        p1g = cast(list[list[float]], p1.genotype)
        p2g = cast(list[list[float]], p2.genotype)

        child_vecs = [
            Crossover.blx_alpha(np.array(v1), np.array(v2), alpha=alpha)[0]
            for v1, v2 in zip(p1g, p2g)
        ]

        child = Individual()
        child.genotype = child_vecs
        child.tags = {"ps": False, "mut": True} 
        child.requires_eval = True
        offspring.append(child)

    return population + offspring

# --- MUTATION ---
def mutation(population: Population,
             mutation_probability: float) -> Population:
    for ind in population:
        if ind.tags.get("mut", False) and not ind.tags.get("ps", False): # Only mutate offspring
            genotype = cast(list[list[float]], ind.genotype)
            mutated = [
                FloatMutator.float_creep(np.array(vec), span=1.0, mutation_probability=mutation_probability)
                for vec in genotype
            ]
            ind.genotype = mutated
            ind.tags["mut"] = False  # Reset mutation tag
    return population


def mutation_crossover(population: Population, lambda_: int, mutation_probability: float, RNG: np.random.Generator) -> Population:
    """Make offspring using mutation"""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    offspring = []

    while len(offspring) < lambda_:
        p = RNG.choice(parents)
        p_genotype = cast(list[list[float]], p.genotype)

        # Mutate
        mutated_vecs = [
            FloatMutator.float_creep(np.array(vec), span=0.1, mutation_probability=mutation_probability)
            for vec in p_genotype
        ]

        child = Individual()
        child.genotype = mutated_vecs
        child.tags = {"ps": False, "mut": True} 
        child.requires_eval = True
        offspring.append(child)

    return offspring