import sys
import os
from typing import Callable

import numpy as np
import pandas as pd
from statistic.euclidean_distance import euclidean_distance
from metrics.rmse import rmse
from data.dataset import Dataset

class KNNRegressor:
    
    def __init__(self,k : int , distance: Callable = euclidean_distance) -> None:
        """_summary_
        Args:
            k (int): number of K examples to be consider
            distance (Callable, optional): Function that calcules the distance between two samples from Dataset. Defaults to euclidean_distance.
        """
        self.k=k
        self.distance=distance
        self.dataset=None

    def fit(self,dataset:Dataset):
        self.dataset=dataset # training dataset
        return self

     
    def _get_closest_label(self, sample):
        distances = self.distance(sample, self.dataset.X)
        k_nearest_neighbor = np.argsort(distances)[:self.k]
        k_nearest_neighbor_label = self.dataset.y[k_nearest_neighbor]

        return np.mean(k_nearest_neighbor_label)
    
    def predict(self, dataset: Dataset):
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)
    
        
    def score(self, dataset: Dataset) -> float:
        prediction = self.predict(dataset)
        return rmse(dataset.y, prediction)