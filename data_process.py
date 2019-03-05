import pandas as pd
import numpy as np
import h5py
import torch
from tqdm import tqdm


class DataProcessor:
    def __init__(self, path, window_size, output_size,
                 split_ratio=None, shuffle=False, random_state=1109):
        """
        You may find the data from this site
        http://cseweb.ucsd.edu/~yaq007/nasdaq100.zip,
        unzip and you can find the padded csv at nasdaq100/small/nasdaq100_padding.csv

        """
        self.data = np.array(pd.read_csv(path))
        self.window_size = window_size
        self.output_size = output_size
        self.split_ratio = split_ratio
        self.shuffle = shuffle
        self.random_state = random_state
        self.X = self.y = None

    def processing(self):
        X = []
        y = []
        for i in tqdm(range(0,self.data.shape[0]-self.output_size),desc='parsing data', ncols=0):
            a = self.data[i:i+self.window_size, :]
            b = self.data[i+self.window_size:i+self.window_size+self.output_size]
            if len(a) == self.window_size and len(b) == self.output_size:
                X.append(a)
                y.append(b)
        X, y = np.array(X), np.array(y)
        self.X = X
        self.y = y

        assert X.shape[0] == y.shape[0] and X.shape[-1] == y.shape[-1], print(X.shape,y.shape)

        if self.shuffle:
            self.shuffle_()

        return self.X, self.y

    def shuffle_(self):

        if self.X is None and self.y is None:
            raise ValueError('did not process the data')

        r = np.random.RandomState(self.random_state)
        perm = r.permutation(self.X.shape[0])
        self.X, self.y = self.X[perm], self.y[perm]
        return self.X, self.y

    def split(self):
        if self.X is None and self.y is None:
            raise ValueError('did not process the data')

        self.shuffle_()
        train_X = self.X[:self.X.shape[0] * self.split_ratio]
        train_y = self.y[:self.X.shape[0] * self.split_ratio]

        test_X = self.X[self.X.shape[0] * self.split_ratio:]
        test_y = self.y[self.X.shape[0] * self.split_ratio:]

        return train_X, train_y, test_X, test_y

    def save(self,path):
        if self.X is None and self.y is None:
            raise ValueError('did not process the data')

        with h5py.File(path, 'w') as f:
            f['X'] = self.X
            f['y'] = self.y

    def load(self, path):
        with h5py.File(path,'r') as f:
            self.X = f['X'][:]
            self.y = f['y'][:]

        return self.X, self.y

