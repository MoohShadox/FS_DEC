import numpy as np
from sklearn.ensemble import RandomForestClassifier

from data_loader.loadGerman import loadGerman
from feature_selection.evaluation.Classification_Evaluation import Classification_Evaluator
from pref.utility.Additive_Utility_Model import Additive_Utility_Model
from pref.utility.Fishburn_Additive_Utility_Model import Fishburn_Utility_Model
from pref.utility.Preferences import sampleSubset, getRelationFromPerformances
from pref.utility.Relations import pareto_dominance


class Wrapper_Model:

    def __init__(self,X, y,  *args, **kwargs):
        self.evaluation = kwargs["evaluation"] if "evaluation" in kwargs else None
        self.X = X
        self.y = y
        self.features = list(range(self.X.shape[1]))
        self.evaluated_subset = {}
        self.n_evaluations = 0
        self.stats = {}


    def best_elements(self, aggregator, k=1):
        s = sorted(self.evaluated_subset.items(), key=lambda x:aggregator(x[1]), reverse=True)
        return [aggregator(v[1]) for v in  s[:k]]

    def save_stats(self, aggregator, k = 10):
        self.stats["evaluations"] = self.stats.get("evaluations", []) + [self.n_evaluations]
        self.stats["mean"] = self.stats.get("mean", []) + [self.best_elements(aggregator, k).mean()]


    def nextSubset(self,subset, *args, **kwargs):
        subset = sampleSubset(self.features)
        return subset

    def testStop(self):
        pass

    def initialSubset(self):
        subset = sampleSubset(self.features)
        return subset

    def evaluateSubset(self, subset, *args, **kwargs):
        if not (subset in self.evaluated_subset):
            self.n_evaluations += 1
            self.evaluated_subset[subset] = self.evaluation(subset, *args, **kwargs)
        return self.evaluated_subset[subset]


    def run(self, max_step = 1000):
        subset = self.initialSubset()
        for step in range(max_step):
            subset = self.nextSubset(subset)
            self.evaluateSubset(subset)
            if(step == max_step):
                break

    def printResults(self):
        for k in self.evaluated_subset:
            print(k," : ", self.evaluated_subset[k])

    def getPreferences(self):
        R = getRelationFromPerformances(self.evaluated_subset, pareto_dominance)
        return R



if __name__ == "__main__":
    X,y = loadGerman()
    evaluator = Classification_Evaluator(X,y, cls=RandomForestClassifier())
    wm = Wrapper_Model(X, y, sampler=sampleSubset, evaluation=evaluator.evaluate)
    wm.run(max_step=30)
    wm.printResults()
