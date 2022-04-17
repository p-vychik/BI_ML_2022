import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    """
    YOUR CODE IS HERE
    """
    tp = np.sum((y_pred == y_true) & (y_true == 0))
    tn = np.sum((y_pred == y_true) & (y_true == 1))
    fn = np.sum((y_pred != y_true) & (y_true == 0))
    fp = np.sum((y_pred != y_true) & (y_true == 1))
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precison = 0
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0
    try:
        f1 = 2 * (precision * recall) / (precision + recall)
    except ZeroDivisionError:
        f1 = 0
    accuracy = (tp + tn) / len(y_true)
    return precision, recall, f1, accuracy
    
    pass


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    try:
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0,1]
        r_sq = corr**2
    except ZeroDivisionError:
        r_sq = 'zero division error'
    return r_sq
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    try:
        mse = np.sum((y_true - y_pred) ** 2) / len(y_true)
    except ZeroDivisionError:
        mse = 'zero division error'
    return mse
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    try:
        mae = np.sum(np.abs(y_true - y_pred)) / len(y_true)
    except ZeroDivisionError:
        mae = 'zero division error'
    return mae
    pass
    
