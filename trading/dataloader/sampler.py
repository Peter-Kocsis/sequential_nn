import matplotlib.pyplot as plt
import torch
from torch.utils.data import Sampler

from trading.dataloader.sequential_dataset import SequentialDataset
from trading.dataloader.stock_dataset import StockHistoryDataset
from trading.dataloader.temperature_dataset import TemperatureDataset


class SequentialDataBatchSampler(Sampler):
    def __init__(
        self,
        data_source: SequentialDataset,
        output_sequence_length,
        num_of_batches=1,
        batch_size=1,
        input_sequence_length: int = None,
        **kwargs
    ):
        super().__init__(data_source)
        self.data_source = data_source
        self.output_sequence_length = output_sequence_length
        self.num_of_batches = num_of_batches
        self.batch_size = batch_size
        self.num_data_points = len(self.data_source)

        self.min_input_sequence_length = (
            input_sequence_length if input_sequence_length is not None else 1
        )
        self.max_sequence_length = (
            input_sequence_length
            if input_sequence_length is not None
            else self.num_data_points - self.output_sequence_length - self.batch_size
        )

        self._check_input()

    def _check_input(self):
        assert (
            self.output_sequence_length > 0
        ), "Output sequence length must be positive"
        assert (
            self.min_input_sequence_length > 0
        ), "Minimal input sequence length must be positive"
        assert (
            self.max_sequence_length > 0
        ), "Maximal input sequence length must be positive"
        assert self.num_of_batches > 0, "Number of batches must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert (
            self.num_data_points
            - self.max_sequence_length
            - self.output_sequence_length
            >= self.batch_size
        ), "Minimal input sequence length is too small, not enough data"

    def __iter__(self):
        """
        Sample input_sequence_length, sample num_of_batches sequences with the given length and iterate over them
        """

        batch_input_sequence_length = torch.randint(
            low=self.min_input_sequence_length,
            high=self.max_sequence_length + 1,
            size=(self.num_of_batches,),
        )

        for batch_idx in range(self.num_of_batches):
            input_sequence_length = batch_input_sequence_length[batch_idx]

            batch_start_idxs = torch.randint(
                low=0,
                high=self.num_data_points
                - input_sequence_length
                - self.output_sequence_length
                + 1,
                size=(self.batch_size,),
            )

            batch_input_idxs = (
                torch.arange(input_sequence_length)[None, :] + batch_start_idxs[:, None]
            )
            batch_output_idxs = (
                torch.arange(self.output_sequence_length)[None, :]
                + batch_start_idxs[:, None]
                + input_sequence_length
            )

            # X_train_batch = self.features[batch_input_idxs]
            # y_train_batch = self.features[batch_output_idxs]
            #
            # X_train_meta = self.data[batch_input_idxs]
            # y_train_meta = self.data[batch_output_idxs]
            yield [
                (batch_input_idxs[data_idx], batch_output_idxs[data_idx])
                for data_idx in range(self.batch_size)
            ]
            # yield X_train_batch, y_train_batch, X_train_meta, y_train_meta

    def __len__(self):
        return self.num_of_batches

    @staticmethod
    def plot(X_train_batch, y_train_batch):
        raise NotImplementedError()


class StockHistoryBatchSampler(SequentialDataBatchSampler):
    @staticmethod
    def plot(X_train_batch, y_train_batch):
        batch_size = X_train_batch.shape[0]
        figure, axes = plt.subplots(nrows=batch_size, ncols=1, dpi=200)

        if batch_size == 1:
            axes = [axes]

        for data_idx in range(batch_size):
            ax = axes[data_idx]
            X_data = StockHistoryDataset.torch_as_pandas(
                X_train_batch[data_idx], StockHistoryDataset.HEADERS, "Date"
            )
            y_data = StockHistoryDataset.torch_as_pandas(
                y_train_batch[data_idx], StockHistoryDataset.HEADERS, "Date"
            )

            StockHistoryDataset.draw(
                X_data,
                ax=ax,
                custom_openkwargs={"label": None},
                custom_closekwargs={"label": None},
                custom_rangekwargs={"label": "Input"},
            )

            last_x = X_data.tail()["Date"]
            ymin = min(X_data["Open"])
            ymax = max(X_data["Open"])
            ax.vlines(
                last_x,
                color="blue",
                linestyles="dashed",
                ymin=ymin,
                ymax=ymax,
                linewidth=2,
            )

            StockHistoryDataset.draw(
                y_data,
                ax=ax,
                custom_openkwargs={"color": "blue", "label": None},
                custom_closekwargs={"color": "cyan", "label": None},
                custom_rangekwargs={"color": "lightblue", "label": "GroundTruth"},
            )
        plt.show()


class TemperatureBatchSampler(SequentialDataBatchSampler):
    @staticmethod
    def plot(X_train_batch, y_train_batch):
        batch_size = X_train_batch.shape[0]
        figure, axes = plt.subplots(nrows=batch_size, ncols=1, dpi=200)

        if batch_size == 1:
            axes = [axes]

        for data_idx in range(batch_size):
            ax = axes[data_idx]
            X_data = TemperatureDataset.torch_as_pandas(
                X_train_batch[data_idx], TemperatureDataset.HEADERS, "date"
            )
            y_data = TemperatureDataset.torch_as_pandas(
                y_train_batch[data_idx], TemperatureDataset.HEADERS, "date"
            )

            TemperatureDataset.draw(
                X_data, ax=ax, custom_value_kwargs={"label": "Input"}
            )

            last_x = X_data.tail()["date"]
            ymin = min(X_data["MT_200"])
            ymax = max(X_data["MT_200"])
            ax.vlines(
                last_x,
                color="blue",
                linestyles="dashed",
                ymin=ymin,
                ymax=ymax,
                linewidth=2,
            )

            TemperatureDataset.draw(
                y_data,
                ax=ax,
                custom_value_kwargs={"color": "lightblue", "label": "GroundTruth"},
            )
        plt.show()
