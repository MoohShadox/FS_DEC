from itertools import combinations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from data_loader.loadAustralian import loadAustralian
from data_loader.loadGerman import loadGerman
from feature_selection.evaluation.Classification_Evaluation import Classification_Evaluator
from feature_selection.wrapper_model import Wrapper_Model
from feature_selection.wrapper_nsga2 import Wrapper_NSGA2
from pref.utility.Additive_Utility_Model import Additive_Utility_Model
from pref.utility.Preferences import sampleSubset, compareElementWithList
from pref.utility.Relations import pareto_dominance


class Wrapper_Fishburn(Wrapper_Model):

    def __init__(self, X, y, *args, **kwargs):
        super().__init__(X, y, *args, **kwargs)
        self.k = kwargs["k"] if "k" in kwargs else 3
        self.model = Additive_Utility_Model(self.features, k = self.k)
        self.model.create_params()
        self.initial_evaluations = kwargs["initial_evaluations"] if "initial_evaluations" in kwargs else 1
        self.preferences = []
        self.theta_r = kwargs["theta_r"] if "theta_r" in kwargs else "MV"
        self.s = self.initialSubset()

    def compute_particular_theta(self):
        self.model.optimize(self.theta_r)

    def power_index(self):
        powers = []
        for i in self.features:
            th = self.model.get_params(lambda x: i in x)
            c = 0
            for t in th:
                c += t.solution_value
            powers.append(c)
        powers = np.array(powers)
        powers = (powers - powers.min() + 1e-5) / (powers.max() - powers.min() + 1e-5)
        powers = powers / (powers.sum())
        return powers

    def nextSubset(self,subset, *args, **kwargs):
        self.compute_particular_theta()
        powers = self.power_index()
        v1 = self.model.get_utility_exp(subset).solution_value
        k = np.random.randint(1, len(self.features))
        s2 = tuple(sorted(np.random.choice(self.features, size=k, p=powers)))
        v2 = self.model.get_utility_exp(s2).solution_value
        cpt = 0
        while(v2 <= v1):
            cpt += 1
            k = np.random.randint(1, len(self.features))
            s2 = tuple(sorted(np.random.choice(self.features, size=k, p=powers)))
            v2 = self.model.get_utility_exp(s2).solution_value
            if(cpt == 1000):
                print("did'nt choose a solution")
                break
        return s2

    def evaluateSubset(self, subset, *args, **kwargs):
        if not (subset in self.evaluated_subset):
            self.n_evaluations += 1
            e = self.evaluation(subset, *args, **kwargs)
            self.evaluated_subset[subset] = e
            R = compareElementWithList(subset, e, self.evaluated_subset, relation=pareto_dominance)
            L = []
            for i in R:
                if(not i in self.preferences):
                    self.preferences.append(i)
                    L.append(i)
            self.model.create_polyhedron(L)
        return self.evaluated_subset[subset]


    def initialSubset(self):
        for i in range(self.initial_evaluations):
            s = sampleSubset(self.features)
            self.evaluateSubset(s)
        self.model.build_objectifs()
        return sampleSubset(self.features)

    def run(self, max_step = 1000):
        for i in range(max_step):
            self.s = self.nextSubset(self.s)
            self.evaluateSubset(self.s)
            #print("best precision: ", np.mean(self.best_elements(aggregator=lambda x:x[0], k=10)))
            #print("number of preferences: ", len(self.preferences))
            #print("with a number of evaluations: ", self.n_evaluations)


if __name__ == "__main__":
    X,y = loadAustralian()
    evaluator = Classification_Evaluator(X,y, cls=RandomForestClassifier())
    wm = Wrapper_Fishburn(X, y, sampler=sampleSubset, evaluation=evaluator.evaluate)
    wm.run(max_step=30)
