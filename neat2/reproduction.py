"""
Handles creation of genomes, either from scratch or by sexual or
asexual reproduction from parents.
"""
import copy
import random
from itertools import count

import neat2.NSGA as NSGA
from neat2.config import ConfigParameter, DefaultClassConfig


# 定义用于计算拥挤距离的函数

# TODO: Provide some sort of optional cross-species performance criteria, which
# are then used to control stagnation and possibly the mutation rate
# configuration. This scheme should be adaptive so that species do not evolve
# to become "cautious" and only make very slow progress.


class DefaultReproduction(DefaultClassConfig):
    """
    Implements the default NEAT-python reproduction scheme:
    explicit fitness sharing with fixed-time species stagnation.
    """

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(param_dict,
                                  [ConfigParameter('elitism', int, 0),
                                   ConfigParameter('survival_threshold', float, 0.2),
                                   ConfigParameter('min_species_size', int, 1)])

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.reproduction_config = config
        self.reporters = reporters
        self.genome_indexer = count(1)

    def create_new(self, genome_type, genome_config, num_genomes):
        new_genomes = {}
        for i in range(num_genomes):
            key = next(self.genome_indexer)
            g = genome_type(key)
            g.configure_new(genome_config)
            new_genomes[key] = g
        return new_genomes

    def reproduce(self, config, solution, pop_size, generation, fitness_function):

        solution2 = copy.deepcopy(solution)
        # Generating offsprings
        while (len(solution2) <=2 * pop_size):
            gid = next(self.genome_indexer)
            parent1_id, parent1 = random.choice(list(solution.items()))
            parent2_id, parent2 = random.choice(list(solution.items()))
            child = config.genome_type(gid)
            child.configure_crossover(parent1, parent2, config.genome_config)
            child.mutate(config.genome_config)
            solution2[gid] = child

        fitness_function(list(solution2.items()), config)

        function1_values2 = [g.fitness[0] for (k, g) in solution2.items()]
        function2_values2 = [g.fitness[1] for (k, g) in solution2.items()]

        non_dominated_sorted_solution2 = NSGA.fast_non_dominated_sort(solution2)

        crowding_distance_values2 = []

        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(

                NSGA.crowding_distance(function1_values2[:], function2_values2[:],
                                       non_dominated_sorted_solution2[i][:]))

        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                NSGA.index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = NSGA.sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break
        p_front1 = {list(solution2.items())[i][0]: list(solution2.items())[i][1] for i in
                   non_dominated_sorted_solution2[0]}
        solution = {list(solution2.items())[i][0]: list(solution2.items())[i][1] for i in new_solution}
        
        return solution, p_front1
