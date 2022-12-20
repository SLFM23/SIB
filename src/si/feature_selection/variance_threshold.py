from statistics import variance
import numpy as np

class VarianceThreshold:
    def __init__ (self, threshold):
        self.threshold=threshold
        self.variance=None

    def transform(self, dataset):
        pass

    def fit (self, dataset):
        variance=dataset.get_variance()
        self.variance=variance
        return self

    def transform (dataset):
        mask=self.variance>self.threshold
        X = X [:,mask]
        features = np.array(dataset)


