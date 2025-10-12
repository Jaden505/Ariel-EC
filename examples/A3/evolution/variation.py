from typing import cast
import numpy as np
from ariel.ec.a001 import Individual
from ariel.ec.a000 import FloatMutator
from ariel.ec.a005 import Crossover

type Population = list[Individual]

# --- CREATE ---
def create_individual(RNG: np.random.Generator, body_vector_length: int, 
                      controller_vector_length: int) -> Individual:
    
    genotype_body = [
        RNG.uniform(0, 1, size=body_vector_length).astype(np.float32).tolist()
        for _ in range(3)
    ]
    genotype_controller = [
        RNG.uniform(0, 1, size=controller_vector_length).astype(np.float32).tolist()
        for _ in range(3)
    ]
    
    ind = Individual()
    ind.genotype = {'body': genotype_body, 'controller': genotype_controller}
    ind.tags = {"ps": False, "mut": False}
    ind.requires_eval = True
    return ind

# --- CROSSOVER ---
def crossover_controller(population: Population,
              RNG: np.random.Generator,
              lambda_: int,
              alpha: float) -> Population:
    """BLX-alpha crossover on controller genotype only."""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    offspring = []

    while len(offspring) < lambda_:
        p1, p2 = RNG.choice(parents, size=2, replace=False)
        # Extract only the controller genotype
        p1g = cast(dict, p1.genotype)["controller"]
        p2g = cast(dict, p2.genotype)["controller"]

        child_controller = [
            Crossover.blx_alpha(np.array(v1), np.array(v2), alpha=alpha)[0]
            for v1, v2 in zip(p1g, p2g)
        ]

        child = Individual()
        # Copy the body genotype from one of the parents (e.g., p1)
        child.genotype = {
            "body": cast(dict, p1.genotype)["body"],
            "controller": child_controller
        }
        child.tags = {"ps": False, "mut": True} 
        child.requires_eval = True
        offspring.append(child)

    return parents + offspring

def crossover_body(population: Population, 
           RNG: np.random.Generator,
           lambda_: int) -> Population:
    """Uniform crossover on body genotype only."""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    offspring = []
    
    while len(offspring) < lambda_:
        p1, p2 = RNG.choice(parents, size=2, replace=False)
        # Extract only the body genotype
        p1g = cast(dict, p1.genotype)["body"]
        p2g = cast(dict, p2.genotype)["body"]

        child_body = []
        for vec1, vec2 in zip(p1g, p2g):
            mask = RNG.choice([True, False], size=len(vec1))
            child_vec = np.where(mask, vec1, vec2).astype(np.float32).tolist()
            child_body.append(child_vec)

        child = Individual()
        # Copy the controller genotype from one of the parents (e.g., p1)
        child.genotype = {
            "body": child_body,
            "controller": cast(dict, p1.genotype)["controller"]
        }
        child.tags = {"ps": False, "mut": True} 
        child.requires_eval = True
        offspring.append(child)
    

# --- MUTATION ---
def mutation_controller(population: Population,
             mutation_probability: float) -> Population:
    for ind in population:
        if ind.tags.get("mut", False) and not ind.tags.get("ps", False): # Only mutate offspring
            genotype = cast(dict, ind.genotype)
            mutated_controller = [
                FloatMutator.float_creep(np.array(vec), span=1.0, mutation_probability=mutation_probability)
                for vec in genotype["controller"]
            ]
            genotype["controller"] = mutated_controller
            ind.genotype = genotype
            ind.tags["mut"] = False  # Reset mutation tag
    return population

def mutation_body(population: Population,
         mutation_probability: float) -> Population:
    """Uniform flip mutation on body genotype only. Randomly set a value to 1 - value if mutation ocurs."""
    for ind in population:
        if ind.tags.get("mut", False) and not ind.tags.get("ps", False): # Only mutate offspring
            genotype = cast(dict, ind.genotype)
            mutated_body = []
            for vec in genotype["body"]:
                vec = np.array(vec)
                mask = np.random.rand(len(vec)) < mutation_probability
                vec[mask] = 1.0 - vec[mask]  # Flip mutation
                mutated_body.append(vec.astype(np.float32).tolist())
            genotype["body"] = mutated_body
            ind.genotype = genotype
            ind.tags["mut"] = False  
    return population
