from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from data_loader.loadGerman import loadGerman
from feature_selection.evaluation.Classification_Evaluation import Classification_Evaluator
from feature_selection.wrapper_model import Wrapper_Model
from pref.utility.Preferences import sampleSubset


class Wrapper_Local_Search(Wrapper_Model):

    def __init__(self, X, y, *args, **kwargs):
        super().__init__(X, y, *args, **kwargs)
        self.aggregator = kwargs["aggregator"]

    def initialSubset(self):
        s = sampleSubset(self.features, k=1)
        return s


    def nextSubset(self,subset, *args, **kwargs):
        not_in = [i for i in self.features if i not in subset]
        neighbourhood = {}
        print("current subset: ", subset, " with : ", self.aggregator(self.evaluateSubset(subset)))
        print("number of calls: ", self.n_evaluations)
        for k in not_in:
            s2 = subset + (k,)
            e = self.evaluateSubset(s2)
            neighbourhood[s2] = e
        for s in subset:
            s2 = tuple([i for i in subset if i!= s])
            if(len(s2) == 0):
                continue
            e = self.evaluateSubset(s2)
            neighbourhood[s2] = e

        sneighbourhood = sorted(neighbourhood.items(), key=lambda x:self.aggregator(x[1]), reverse=True)
        print("best neighbour found: ", sneighbourhood[0])
        if(self.aggregator(sneighbourhood[0][1]) > self.aggregator(self.evaluateSubset(subset))):
            return sneighbourhood[0][0]
        else:
            return subset




    def testStop(self):
        pass


if __name__ == "__main__":
    X, y = loadGerman()
    #X,y = make_classification(n_samples=1000, n_informative=10)
    evaluator = Classification_Evaluator(X, y, cls=RandomForestClassifier())
    wls = Wrapper_Local_Search(X,y, evaluation=evaluator.evaluate, aggregator=lambda x:x[0])
    wls.run(max_step=100)