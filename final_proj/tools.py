from abc import ABC
import operator
from functools import reduce
import numpy as np


def product(args):
    return reduce(operator.mul, args, 1)


def get_lengths(shapes):
    """ get vector of lengths (number of elements) of weight matrices """
    return [product(shape) for shape in shapes]


def unpack(w, shapes, lengths=None):
    ''' given a weights vector, build w parts from it '''
    if lengths is None:
        lengths = get_lengths(shapes)
    last_length = 0
    ws = []
    for shape, length in zip(shapes, lengths):
        next_length = last_length + length
        ws.append(w[last_length:next_length].reshape(shape))
        last_length = next_length
    return ws


def init_weights(shapes):
    ''' generate weights vector '''
    lengths = get_lengths(shapes)
    w = np.random.uniform(-1, 1, size=sum(lengths))
    return w, unpack(w, shapes, lengths)


class TimeSeriesForecaster(ABC):
    """
    When it comes to data, here are the naming conventions and formats:
    t = the array of time points (shape: time)
    u = the matrix of inputs (shape: time x dimension)
    d = the target values (shape: time x dimension)
    y = the predicted values (shape: time x dimension)
    """

    def __init__(self, params):
        self.params = params

    @staticmethod
    def rmse(d, y):
        num_samples = d.shape[0]
        return np.sqrt(np.sum((d-y)**2) / num_samples)

    @staticmethod
    def scale_matrix(W):
        if W.shape[1] == 0:
            return W  # no matrix to scale
        return W / np.sqrt(W.shape[1])

    def score(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
