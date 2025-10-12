from ariel.ec.a001 import Individual
from simulation import evolve_simulation
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

type Population = list[Individual]


def evaluate(population: Population, nde: NeuralDevelopmentalEncoding) -> Population:
    for ind in population:
        if ind.requires_eval:
            ind.fitness = evolve_simulation(nde, ind.genotype['body'], ind.genotype['controller'])
            ind.requires_eval = False
    return population
