import copy
from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters


register_matplotlib_converters()


class SequentialDataset(Dataset):
    FEATURES = None
    HEADERS = None
    OUTPUTS = None

    def __init__(self, *args, root_path: str, **kwargs):
        self.root_path = root_path
        self.data_source = None
        self.name = None

        self._features = None
        self._outputs = None
        self._data = None

    @staticmethod
    def draw(data, ax=None, **kwargs):
        raise NotImplementedError()

    def plot(self):
        plt.figure(1, figsize=(len(self) / 300, 10), dpi=200)
        self.draw(self.data_source)
        plt.title(f"{self.name}")
        plt.legend()
        plt.show()

    def pandas_as_torch(self, features):
        data = self.data_source[features]
        date_columns = [
            column_name
            for column_name, is_date in (data.dtypes == "datetime64[ns]").items()
            if is_date
        ]
        data[date_columns] = data[date_columns].astype(int)
        return torch.tensor(data.values, dtype=torch.float32)

    @staticmethod
    def torch_as_pandas(data_tensor, headers, date_headers):
        pandas_data = pd.DataFrame(data_tensor.numpy())
        pandas_data.columns = headers
        pandas_data[date_headers] = pd.to_datetime(pandas_data[date_headers])
        return pandas_data

    def get_range(self, start=0, end=None):
        data_slice = copy.deepcopy(self)
        data_slice.data_source = self.data_source[start:end]

        data_slice._features = None
        data_slice._data = None
        return data_slice

    @property
    def features(self):
        if self._features is None:
            self._features = self.pandas_as_torch(self.FEATURES)
        return self._features

    @property
    def outputs(self):
        if self._outputs is None:
            self._outputs = self.pandas_as_torch(self.OUTPUTS)
        return self._outputs

    @property
    def data(self):
        if self._data is None:
            self._data = self.pandas_as_torch(self.HEADERS)
        return self._data

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return self.data_source.iloc[key]
            # return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            # get the data from direct index
            return self.data_source.iloc[key]
        elif isinstance(key, tuple):
            assert len(key) == 2, "Only tuple with length 2 is supported"
            batch_input_idxs, batch_output_idxs = key
            X_train_batch = self.features[batch_input_idxs]
            y_train_batch = self.outputs[batch_output_idxs]

            X_train_meta = self.data[batch_input_idxs]
            y_train_meta = self.data[batch_output_idxs]
            return X_train_batch, y_train_batch, X_train_meta, y_train_meta
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    def __len__(self):
        return len(self.data_source)
