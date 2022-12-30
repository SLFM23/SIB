import sys
sys.path.insert(0, '/Users/Sergiomendes/Desktop/SIB/SIB/src/si/')

from typing import Callable
import numpy as np
from statistic.f_classification import f_classification
from data.dataset import Dataset

class SelectPercentile:
    def __init__(self,score_function: Callable = f_classification, percentile: float = 0.25):
        self.score_function=score_function
        self.percentile= percentile
        self.F=None
        self.p=None

    def fit (self,dataset):
        self.F,self.p=self.score_function(dataset)
        return self

    def transformer (self,dataset):
        len_features = len(dataset.features)
        percentile = int(len_features * self.percentile)
        idxs = np.argsort(self.F)[-percentile:] # queremos as mehores, com o limite do percentil
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self,dataset):
        self.fit(dataset)
        return self.transformer(dataset)


if __name__ == "__main__":
    a = SelectPercentile(percentile = 0.6)
    dataset = Dataset(X=np.array([[0, 3, 3, 3],
                                  [0, 2, 2, 2],
                                  [0, 1, 1, 1]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    a.fit(dataset)
    b = a.transformer(dataset)
    print(b.features) 