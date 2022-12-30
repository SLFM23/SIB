import numpy as np

def accuracy(y_true,y_pred):
    """ Calculates the error between arguments given using the accuracy formula:
    
                    (VN + VP) / (VN + VP + FP + FN)
    Args:
        y_true (_type_): real values
        Y_pred (_type_): predicted values
    Returns:
        float: Error value between y_true and y_pred
    """

    return np.sum(y_true == y_pred) / len(y_true)