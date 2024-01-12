"""Implements the core evolution algorithm."""
import time
import os
from neat2.math_util import mean
from neat2.reporting import ReporterSet
import shutil
import sys
from utils import create_folder
import pickle

class CompleteExtinctionException(Exception):
    pass


class Population(object):

    def __init__(self, config, initial_state=None):
        self.reporters = ReporterSet()
        self.config = config
        self.reproduction = config.reproduction_type(config.reproduction_config,self.reporters)

        if initial_state is None:
            # Create a population from scratch, then partition into species.
            self.population = self.reproduction.create_new(config.genome_type,
                                                           config.genome_config,
                                                           config.pop_size)
            self.generation = 0
        else:
            self.population, self.generation = initial_state

        self.best_genomes = None

    # def add_reporter(self, reporter):
    #     self.reporters.add(reporter)
    #
    # def remove_reporter(self, reporter):
    #     self.reporters.remove(reporter)

    def run(self, fitness_function, n=None):

        k = 0
        while n is None or k < n:
            # os.mkdir(f"/mnt/data/mhyan/fit_cal")
            k += 1
            print(f"the gen of {k} begin")
            # Create the next generation from the current generation.
            self.population ,self.best_genomes= self.reproduction.reproduce(self.config,self.population,
                                                          self.config.pop_size, self.generation,fitness_function)
            # avg
            sum_f=0
            sum_v=0
            for i,g in enumerate(list(self.best_genomes.items())):
                sum_f+=g[1].fitness[0]
                sum_v+=g[1].fitness[1]
            lenofg=len(self.best_genomes)
            print(f"the average fitness of best front is {sum_f/lenofg,sum_v/lenofg}")
            if k==3:
                for i,g in enumerate((list(self.best_genomes.items()))):
                    path=f"./output_gen0/g{i}"
                    create_folder(path)
                    with open(f'{path}/genome.pkl','wb') as f:
                        pickle.dump(g[1],f)
            self.generation += 1

        return self.best_genomes
