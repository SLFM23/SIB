from typing import Callable
import pandas as pd
import numpy as np

from si.statistics.f_classification import f_classification
from si.data.dataset import Dataset


class SelectKBest:
    def __init__(self,score_function: Callable = f_classification, k: int= 10):
        self.score_function=score_function
        self.k=k
        self.F=None
        self.p=None 

    def fit(self,dataset):
        self.F,self.p=self.score_function(dataset)
        return self


    def transformer (self,dataset):
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.labels)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transformer(dataset)


if __name__ == "__main__":
    a = SelectKBest(k = 3)
    dataset = Dataset(X=np.array([[0, 3, 3, 3],
                                  [0, 2, 2, 2],
                                  [0, 1, 1, 1]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a.fit(dataset)
    b = a.transformer(dataset)
    print(b.features) 