import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data_loader.loadAustralian import loadAustralian
from data_loader.loadGerman import loadGerman
from feature_selection.evaluation.Classification_Evaluation import Classification_Evaluator
from feature_selection.wrapper_model import Wrapper_Model
from pref.utility.Preferences import sampleSubset, getRelationFromPerformances

from deap import base, tools, creator, algorithms


def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]


class Wrapper_NSGA2(Wrapper_Model):

    def __init__(self, X, y, *args, **kwargs):
        super().__init__(X, y, *args, **kwargs)
        self.P = kwargs["P"] if "P" in kwargs else 12
        self.MU = kwargs["MU"] if "MU" in kwargs else 5
        self.CXPB, self.MUTPB = 0.3, 0.2

        creator.create("FitnessMin", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)


        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_bool", random.randint, 0, 1)
        # Structure initializers
        self.toolbox.register("individual", tools.initRepeat, creator.Individual,
                         self.toolbox.attr_bool, len(self.features))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)


        ref_points = tools.uniform_reference_points(2, self.P)

        self.toolbox.register("evaluate", self.evaluateSubset)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        self.toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

        # Initialize statistics object
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        self.population = self.toolbox.population(n=self.MU)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
        fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Begin the generational process


    def nextSubset(self,subset, *args, **kwargs):
        subset = sampleSubset(self.features)
        return subset

    def evaluateSubset(self, bin_subset, *args, **kwargs):
        subset = np.array(bin_subset)
        subset = tuple(np.where(subset == 1)[0])
        if not (subset in self.evaluated_subset):
            self.n_evaluations += 1
            self.evaluated_subset[subset] = self.evaluation(subset, *args, **kwargs)
        return self.evaluated_subset[subset]


    def initialSubset(self):
        subset = sampleSubset(self.features)
        return subset


    def run(self, max_step = 1000):
        for gen in range(0, max_step):
            offspring = algorithms.varAnd(self.population, self.toolbox, self.CXPB, self.MUTPB)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            self.population = self.toolbox.select(self.population + offspring, self.MU)
            print("best precision: ", self.best_elements(aggregator=lambda x:x[0], k=1))
            print("with a number of evaluations: ", self.n_evaluations)


if __name__ == "__main__":
    X,y = loadAustralian()
    evaluator = Classification_Evaluator(X,y, cls=RandomForestClassifier())
    wm = Wrapper_NSGA2(X, y, sampler=sampleSubset, evaluation=evaluator.evaluate)
    wm.run(max_step=1)
    wm.printResults()
