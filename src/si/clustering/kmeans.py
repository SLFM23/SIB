from typing import Callable
import numpy as np
from sklearn.metrics import euclidean_distances

def __init__ (self, k:int, max_inter:int=1000, distance:Callable=euclidean_distance):
    self.k = k
    self.max_iter=max