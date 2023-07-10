from __future__ import print_function, division, absolute_import, unicode_literals
import torch

class BaseDataProvider(object):

    def _load_data_and_label(self):
        data, label, path = self._next_data()
        data, label = self._augment_data(data, label)

        #data = data.transpose(2, 0, 1).astype(float)
        #labels = label.transpose(2, 0, 1).astype(float)
        nd = data.shape[1]
        nw = data.shape[2]
        nh = data.shape[3]
        return path, data.reshape(1, 1, nd, nw, nh), label.reshape(1, 1, nd, nw, nh)


    def __call__(self, n):
        P = []
        for i in range(n):
            path, data, labels = self._load_data_and_label()
        X = data
        Y = labels
        P.append(path)

        return X, Y, P