from ariel.ec.a001 import Individual
from ariel.simulation.controllers.controller import Controller

from collections.abc import Callable

type Population = list[Individual]


def evaluate(population: Population, fitness_func: Callable, controller: Controller) -> Population:
    for ind in population:
        if ind.requires_eval:
            ind.fitness = fitness_func(ind.genotype, controller)
            ind.requires_eval = False
    return population
