import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # implement metrics!

    tp = np.logical_and(prediction, ground_truth).sum()
    tn = np.logical_and(np.logical_not(prediction),
                        np.logical_not(ground_truth)).sum()
    precision = tp/prediction.sum()
    recall = tp/ground_truth.sum()
    accuracy = (tp+tn)/prediction.size
    f1 = 2*precision*recall/(precision+recall)

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return np.sum(prediction == ground_truth)/ground_truth.size
