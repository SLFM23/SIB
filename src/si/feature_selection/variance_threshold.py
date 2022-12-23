import numpy as np
from si.data.dataset import Dataset

class VarianceThreshold:
    def __init__ (self, threshold = 0.0):
        self.threshold=threshold
        self.variance=None

    def transform(self, dataset):
        mask= self.variance>self.threshold
        newX=dataset.X[:,mask]
        features= np.array(dataset.features)[mask]
        return Dataset(X=newX,y=dataset.y,features=list(features),label=None)

    def fit (self, dataset):
        variance=dataset.get_variance()
        self.variance=variance
        return self

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == "__main__":
    dataset = Dataset(X=np.array([[0, 3, 3, 3],
                                  [0, 2, 2, 2],
                                  [0, 1, 1, 1]]),
                      y=np.array([0, 1, 0]),
                      features=["f1", "f2", "f3", "f4"],
                      label="y")
    
    b = VarianceThreshold(3)
    b = b.fit_transform(dataset)
    print(b.features)
