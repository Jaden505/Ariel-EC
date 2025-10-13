from ariel.ec.a001 import Individual
from simulation import evolve_simulation
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from neural_decoder import ControllerDecoder

type Population = list[Individual]


def evaluate(population: Population, nde: NeuralDevelopmentalEncoding, control_decoder: ControllerDecoder) -> Population:
    for ind in population:
        if ind.requires_eval:
            ind.fitness = evolve_simulation(nde, ind.genotype['body'], ind.genotype['controller'], control_decoder)
            ind.requires_eval = False
    return population
