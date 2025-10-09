# Standard libraries
from typing import TYPE_CHECKING, cast
import numpy as np

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.a005 import Crossover
from ariel.ec.a000 import FloatMutator

type Population = list[Individual]

def crossover(population: Population, RNG: np.random.Generator, 
              lambda_: float, alpha: float, evolve_morphology: bool) -> Population:
    """BLX-alpha crossover creates lambda_ offspring"""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    offspring = []

    while len(offspring) < lambda_:
        p1, p2 = RNG.choice(parents, size=2, replace=False)
        p1g = p1.genotype
        p2g = p2.genotype

        # BLX crossover for each morphology vector
        if evolve_morphology:
            child_morph = [
                Crossover.blx_alpha(a, b, alpha=alpha)[0]
                for a, b in zip(p1g["morphology"], p2g["morphology"])
            ]
        else:
            child_morph = cast(list, p1g["morphology"])  # No crossover, just copy parent 1

        # BLX crossover for controller
        child_ctrl, _ = Crossover.blx_alpha(p1g["controller"], p2g["controller"], alpha=alpha)

        child = Individual()
        child.genotype = {"morphology": child_morph, "controller": child_ctrl}
        child.tags = {"ps": False, "mut": True}
        child.requires_eval = True

        offspring.append(child)
        
    return offspring



def mutation(population: Population, mutation_probability: float, evolve_morphology: bool = False) -> Population:
    """Float creep mutation"""
    for ind in population:
        if ind.tags.get("mut", False):
            geno = ind.genotype

            if evolve_morphology:
                geno["morphology"] = [
                    FloatMutator.float_creep(v, span=1, mutation_probability=mutation_probability)
                    for v in geno["morphology"]
                ]

            geno["controller"] = FloatMutator.float_creep(
                geno["controller"],
                span=1,
                mutation_probability=mutation_probability
            )
    return population



def create_individual(RNG: np.random.Generator, controller_size: int, best_individual=None) -> Individual:
    input_size = 16  # Replace with actual or dynamic len(data.qpos)
    hidden_size = 8
    output_size = 8  # Replace with actual or dynamic model.nu

    w1 = RNG.normal(0, 0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(0, 0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(0, 0.5, size=(hidden_size, output_size))

    ind = Individual()
    ind.genotype = {
        "morphology": [RNG.uniform(0, 1, size=64).tolist() for _ in range(3)],
        "controller": np.concatenate([w1.flatten(), w2.flatten(), w3.flatten()]).tolist(),
    }
    ind.tags = {"ps": False, "mut": False}
    ind.requires_eval = True
    return ind

