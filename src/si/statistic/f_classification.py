import pandas as pd
import numpy as np
from scipy import stats

from si.data.dataset import Dataset
from typing import Tuple, Union


def f_classification(dataset:Dataset):
    classes = dataset.get_classes()
    groups = [dataset.X[dataset.y == c] for c in classes]
    F,p = stats.f_oneway(*groups)
    return F,p