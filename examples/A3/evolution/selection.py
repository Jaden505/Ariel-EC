from ariel.ec.a001 import Individual
import random

type Population = list[Individual]

def parent_selection(population: Population, mu:int) -> Population:
    """Uniform selection"""
    random.shuffle(population)
    selected = population[:mu]
    
    for ind in selected:
        ind.tags["ps"] = True  # Mark as selected for parenthood
    
    return selected


def survivor_selection(population: Population, mu:int) -> Population:
    """Elitism"""
    non_parents = [ind for ind in population if not ind.tags.get("ps", False)]
    sorted_population = sorted(non_parents, key=lambda ind: ind.fitness, reverse=True)
    
    # Select the top mu individuals
    selected = sorted_population[:mu]
    
    # Reset tags for the next generation
    for ind in selected:
        ind.tags = {"ps": False, "mut": False}
    
    return selected