"""Minimal DataSequence shim so the cached `wordseqs.joblib` can be unpickled.

The huge-data wordseqs were pickled referencing `ridge_utils.DataSequence.DataSequence`.
We only need the attributes used downstream (`data`, `data_times`, `tr_times`).
"""
import numpy as np


class DataSequence(object):
    def __init__(self, data, split_inds, data_times=None, tr_times=None):
        self.data = data
        self.split_inds = split_inds
        self.data_times = data_times
        self.tr_times = tr_times

    def chunks(self):
        return np.split(self.data, self.split_inds)
