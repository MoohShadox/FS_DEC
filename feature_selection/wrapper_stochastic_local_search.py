import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from data_loader.loadGerman import loadGerman
from feature_selection.evaluation.Classification_Evaluation import Classification_Evaluator
from feature_selection.wrapper_model import Wrapper_Model
from pref.utility.Preferences import sampleSubset


class Wrapper_Stochastic_Local_Search(Wrapper_Model):

    def __init__(self, X, y, *args, **kwargs):
        super().__init__(X, y, *args, **kwargs)
        self.aggregator = kwargs["aggregator"]
        self.probability_ponderator = kwargs["probability_ponderator"] if "probability_ponderator" in kwargs else lambda x:x / x.sum()

    def initialSubset(self):
        s = sampleSubset(self.features, k=1)
        return s


    def nextSubset(self,subset, *args, **kwargs):
        not_in = [i for i in self.features if i not in subset]
        neighbourhood = {}
        print("Current subset: ", subset, " : ", self.aggregator(self.evaluateSubset(subset)))
        print("number of calls: ", self.n_evaluations)
        print("=============================")
        for k in not_in:
            s2 = subset + (k,)
            e = self.evaluateSubset(s2)
            neighbourhood[s2] = e
            print("Evaluating: ", s2, " : ", e)
        for s in subset:
            s2 = tuple([i for i in subset if i!= s])
            if(len(s2) == 0):
                continue
            e = self.evaluateSubset(s2)
            neighbourhood[s2] = e
            print("Evaluating: ", s2, " : ", e)

        print("=============================")
        neighbours_values = np.array([self.aggregator(i) for i in neighbourhood.values()])
        neighbours_values = self.probability_ponderator(neighbours_values)
        neighbours_keys = neighbourhood.keys()
        choosed = np.random.choice(np.arange(len(neighbours_keys)), p = neighbours_values)
        return (list(neighbours_keys)[choosed])


    def testStop(self):
        pass


if __name__ == "__main__":
    X, y = loadGerman()
    #X,y = make_classification(n_samples=1000, n_informative=10)
    evaluator = Classification_Evaluator(X, y, cls=RandomForestClassifier())
    arr = np.array([1,2,3])
    wls = Wrapper_Stochastic_Local_Search(X,y, evaluation=evaluator.evaluate, aggregator=lambda x:x[0], probability_ponderator=lambda x:np.exp(x) / np.sum(np.exp(x)))
    wls.run(max_step=100)