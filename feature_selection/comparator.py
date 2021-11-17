import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from feature_selection.evaluation.Classification_Evaluation import Classification_Evaluator
from feature_selection.wrapper_fishburn import Wrapper_Fishburn
from feature_selection.wrapper_local_search import Wrapper_Local_Search
from feature_selection.wrapper_model import Wrapper_Model
from feature_selection.wrapper_nsga2 import Wrapper_NSGA2
from pref.utility.Preferences import sampleSubset


def main_comparators(classifiers, wrappers, best_k = 1,aggregator = lambda x:x[0], dataloader = None, nb_datasets = 50):
    datas = {}
    if not dataloader:
        dataloader = make_classification
    for c in classifiers:
        for dt in range(nb_datasets):
            X,y = dataloader(n_samples=500, n_features=15, n_informative=4)
            evaluator = Classification_Evaluator(X,y, cls=c())
            for wrapper in wrappers:
                print("Wrapper: ", str(wrapper.__name__))
                w = wrapper(X,y,sampler=sampleSubset, evaluation=evaluator.evaluate, aggregator=lambda x:x[0])
                for i in range(100):
                    try:
                        w.run(max_step = 1)
                    except(Exception):
                        print("error for model: ", wrapper)
                        break
                    print("runned: ", i)
                    datas["evaluations"] = datas.get("evaluations", []) + [w.n_evaluations]
                    datas["method"] = datas.get("method", []) + [str(w.__class__.__name__)]
                    datas["classifier"] = datas.get("classifier", []) + [str(c.__class__.__name__)]
                    datas["dataset"] = datas.get("dataset", []) + [str(dt)]
                    datas["performance"] = datas.get("performance", []) + [np.array([w.best_elements(aggregator=lambda x:x[0], k=1)]).mean()]

                df = pd.DataFrame(datas)
                df.to_csv("results2.csv")





if __name__ == "__main__":
    main_comparators([RandomForestClassifier, DecisionTreeClassifier, LogisticRegression], wrappers=[Wrapper_NSGA2 ,Wrapper_Model, Wrapper_Fishburn])