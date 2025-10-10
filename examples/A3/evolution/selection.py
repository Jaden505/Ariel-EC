from ariel.ec.a001 import Individual
import random

type Population = list[Individual]

def parent_selection(population: Population, mu:int) -> Population:
    """Uniform selection"""
    random.shuffle(population)
    selected = population[:mu]
    
    for ind in population:
        if ind in selected:
            ind.tags["ps"] = True
        else:
            ind.tags["ps"] = False
                                                
    return population


def survivor_selection(population: Population, mu: int) -> Population:
    """(μ, λ) survivor selection — only offspring survive."""
    offspring = [ind for ind in population if not ind.tags.get("ps", False)]

    sorted_population = sorted(offspring, key=lambda ind: ind.fitness, reverse=True)

    # Select top μ survivors
    selected = sorted_population[:mu]
    not_selected = sorted_population[mu:]

    for ind in selected:
        ind.alive = True
        ind.tags = {"ps": False, "mut": False}

    for ind in not_selected:
        ind.alive = False

    return population
