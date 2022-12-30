import numpy as np
from data.dataset import Dataset

def train_test_split(dataset:Dataset,test_size: float = 0.2, random_state: int = 40):
    np.random.seed(random_state)
    
    len_samples = dataset.shape()[0]
    len_test = int(test_size * len_samples)
    permutations = np.random.permutation(len_samples)
    test_split = permutations[:len_test]
    train_split = permutations[len_test:]
    train = Dataset(dataset.X[train_split], dataset.y[train_split], features = dataset.features,
                    label = dataset.label)

    test = Dataset(dataset.X[test_split], dataset.y[test_split], features = dataset.features,
                   label = dataset.label)
    return train,test