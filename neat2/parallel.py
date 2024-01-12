"""
Runs evaluation functions in parallel subprocesses
in order to evaluate multiple genomes at once.
"""
from multiprocessing import Pool


class ParallelEvaluator(object):
    def __init__(self, num_workers, eval_function, timeout=None, maxtasksperchild=None):
        """
        eval_function should take one argument, a tuple of (genome object, config object),
        and return a single float (the genome's fitness).
        """
        # Todo multifitness？？
        self.eval_function = eval_function
        self.timeout = timeout
        self.pool = Pool(processes=num_workers, maxtasksperchild=maxtasksperchild)

    def __del__(self):
        self.pool.close()
        self.pool.join()
        self.pool.terminate()

    def evaluate(self, genomes, config):
        jobs = []
        for i,(ignored_genome_id, genome) in enumerate(genomes):
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))

        # Todo need to change！！
        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            f,u=job.get(timeout=self.timeout)
            genome.fitness = f
            genome.u=u
