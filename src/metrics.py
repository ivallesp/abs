import numpy as np


def accuracy(y_true, y_pred):
    pred_labels = y_pred.argmax(axis=1)
    hits = y_true == pred_labels
    assert len(hits.shape) == 1
    return hits.mean()


def crossentropy(y_true, y_pred, eps=1e-9):
    mask = np.eye(y_pred.shape[1])[y_true]
    p = np.sum(y_pred * mask, axis=1)
    return -np.log(np.clip(p, eps, 1 - eps)).mean()
