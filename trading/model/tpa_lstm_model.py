import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from trading.dataloader.sampler import SequentialDataBatchSampler
from trading.dataloader.sequential_dataset import SequentialDataset
from trading.model.network.scaler import MinMaxScaler
from trading.model.network.tpa_lstm_nn import TPALSTM


class TPA_LSTM_Model(pl.LightningModule):
    def __init__(self, hparams, sequential_dataset=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = {}
        # self.feature_scaler = MaxScaler(**hparams.get('feature_scaler', {}))
        # self.output_scaler = MaxScaler(**hparams.get('output_scaler', {}))
        self.feature_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()
        self.network = TPALSTM(**hparams)

        self.hparams = hparams
        self.name = "TPA_LSTM model"

        self.sequential_dataset = sequential_dataset
        self.visualization_dataset = None
        self.draw_params = hparams.get("draw_params", None)

    def forward(self, input):
        self.output_scaler.fit(input[:, :, 0])
        scaled_input = self.feature_scaler.forward(input)

        # scaled_input = input
        network_output = self.network(scaled_input)
        # output = network_output
        output = self.output_scaler.inverse(network_output)
        return output

    def draw(
        self, input: SequentialDataset, ground_truth: SequentialDataset = None, ax=None
    ):

        if ax is None:
            ax = plt

        input_data = torch.squeeze(torch.unsqueeze(input.features, dim=0), dim=2)
        prediction = torch.squeeze(self.forward(input_data.to(self.device)))

        prediction_length = len(prediction)

        input.draw(
            input.data_source,
            ax=ax,
            custom_draw_kwargs=self.draw_params["input_kwargs"],
        )

        last_x = input[-1]["Date"]
        last_last_x = input[-2]["Date"]
        ax.axvline(last_x, color="blue", linewidth=2)

        if ground_truth is not None:
            ground_truth.draw(
                ground_truth.data_source,
                ax=ax,
                custom_draw_kwargs=self.draw_params["ground_truth_kwargs"],
            )
            prediction_x = ground_truth.data_source["Date"]
        else:
            prediction_x = pd.Series(
                pd.date_range(
                    last_x, periods=prediction_length, freq=last_x - last_last_x
                )
            )
        ax.plot(
            prediction_x, prediction.detach().cpu(), color="red", label="Prediction"
        )

    def plot(self, input: SequentialDataset, ground_truth: SequentialDataset = None):
        plt.figure(
            1,
            figsize=(
                (
                    self.hparams["input_sequence_length"]
                    + self.hparams["output_sequence_length"]
                )
                / 20,
                4,
            ),
            dpi=200,
        )

        self.draw(input, ground_truth)

        plt.title(f"Model prediction - {self.name}")
        plt.legend()
        plt.show()

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def log_histogram(self):
        # iterating through all parameters
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)

    @torch.no_grad()
    def create_confusion_matrices(self, loader, future_days):
        all_preds = {future_day: torch.tensor([]) for future_day in future_days}
        all_reals = {future_day: torch.tensor([]) for future_day in future_days}
        for batch in loader:
            X_train_batch, y_train_batch, X_train_meta, y_train_meta = batch
            y_train_batch.squeeze_()

            # Load X, y to device!
            X_train_batch, y_train_batch = (
                X_train_batch.to(self.device),
                y_train_batch.to(self.device),
            )
            last_values = X_train_batch[:, -1, 0]

            y_pred = self.forward(X_train_batch)

            for future_day in future_days:
                pred_labels = (y_pred[:, future_day - 1] - last_values) < 0
                real_labels = (y_train_batch[:, future_day - 1] - last_values) < 0

                all_preds[future_day] = torch.cat(
                    (all_preds[future_day], pred_labels.cpu()), dim=0
                )
                all_reals[future_day] = torch.cat(
                    (all_reals[future_day], real_labels.cpu()), dim=0
                )

        return {
            future_day: confusion_matrix(
                all_reals[future_day].cpu(), all_preds[future_day].cpu()
            )
            for future_day in future_days
        }

    def plot_confusion_matrix(self, confusions, title):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

        fig.suptitle(title)

        for (future_name, cm), ax in zip(confusions.items(), axs.flatten()):
            ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(f"{future_name}_day")

            class_names = ["Up", "Down"]
            tick_marks = np.arange(len(class_names))
            plt.setp(
                ax,
                xticks=tick_marks,
                xticklabels=class_names,
                yticks=tick_marks,
                yticklabels=class_names,
            )

            # Normalize the confusion matrix.
            cm = np.around(
                cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2
            )

            # Use white text if squares are dark; otherwise black.
            threshold = 0.5
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                color = "white" if cm[i, j] > threshold else "black"
                ax.text(j, i, cm[i, j], horizontalalignment="center", color=color)
                ax.set_ylabel("True label")
                ax.set_xlabel("Predicted label")

        plt.tight_layout()
        fig.canvas.draw()

        self.logger.experiment.add_figure(title, fig, global_step=self.current_epoch)
        fig.clf()

    def plot_prediction(self, name, visualization_dataset):
        fig, axs = plt.subplots(
            nrows=len(visualization_dataset),
            ncols=1,
            figsize=(
                (
                    self.hparams["input_sequence_length"]
                    + self.hparams["output_sequence_length"]
                )
                / 20,
                4,
            ),
            dpi=200,
        )
        for ax, visualization_data in zip(axs, visualization_dataset):
            self.draw(
                visualization_data.get_range(
                    0, -self.hparams["output_sequence_length"]
                ),
                visualization_data.get_range(
                    -self.hparams["output_sequence_length"], None
                ),
                ax=ax,
            )
            ax.legend(loc="upper left")

        fig.suptitle(f"Model prediction - {self.name}")

        fig.canvas.draw()

        self.logger.experiment.add_figure(name, fig, global_step=self.current_epoch)
        fig.clf()

    def general_step(self, batch, batch_idx, mode):
        X_train_batch, y_train_batch, X_train_meta, y_train_meta = batch
        y_train_batch.squeeze_()

        # Load X, y to device!
        X_train_batch, y_train_batch = X_train_batch.to(self.device), y_train_batch.to(
            self.device
        )

        # forward pass
        y_pred = self.forward(X_train_batch)

        # loss
        loss = F.mse_loss(y_pred, y_train_batch)

        # accuracy
        acc = self.calculate_accuracy(y_pred, y_train_batch)

        tensorboard_logs = {f"{mode}_loss": loss, f"{mode}_acc": acc}
        return {"loss": loss, **tensorboard_logs, "log": tensorboard_logs}

    def calculate_accuracy(self, predict, real):
        return 1 - torch.sqrt(torch.mean(torch.square((real - predict) / real)))

    def general_epoch_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        losses = torch.stack([x[mode + "_loss"] for x in outputs])
        acces = torch.stack([x[mode + "_acc"] for x in outputs])

        self.plot_prediction(mode, self.visualization_dataset[mode])

        dataloader = self.get_dataloader(mode)
        confusions = self.create_confusion_matrices(dataloader, [1, 5, 10, 30])
        self.plot_confusion_matrix(confusions, title=f"{mode}_confusion")

        tensorboard_logs = {f"{mode}_loss": losses.mean(), f"{mode}_acc": acces.mean()}
        return {**tensorboard_logs, "log": tensorboard_logs}

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")

    def training_epoch_end(self, outputs):
        return self.general_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "valid")

    def validation_epoch_end(self, outputs):
        return self.general_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def test_epoch_end(self, outputs):
        self.log_histogram()
        return self.general_epoch_end(outputs, "test")

    def prepare_data(self):

        dataset = self.sequential_dataset

        # self.feature_scaler.update_params(dataset.features)
        # self.output_scaler.update_params(dataset.outputs)
        #
        # self.hparams['feature_scaler'] = {'scaler_max': self.feature_scaler.max.numpy()}
        # self.hparams['output_scaler'] = {'scaler_max': self.output_scaler.max.numpy()}

        train_ratio = 0.6
        valid_ratio = 0.2
        last_train_idx = int(len(dataset) * train_ratio)
        last_valid_idx = int(len(dataset) * valid_ratio) + last_train_idx
        train_data = dataset.get_range(start=0, end=last_train_idx)
        valid_data = dataset.get_range(start=last_train_idx, end=last_valid_idx)
        test_data = dataset.get_range(start=last_valid_idx)

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["valid"], self.dataset["test"] = (
            train_data,
            valid_data,
            test_data,
        )

        visualization_dataset_length = (
            self.hparams["input_sequence_length"]
            + self.hparams["output_sequence_length"]
        )
        self.visualization_dataset = {
            "train": [
                train_data.get_range(0, visualization_dataset_length),  # Beginning
                train_data.get_range(-visualization_dataset_length, None),  # End
            ],
            "valid": [
                valid_data.get_range(0, visualization_dataset_length),  # Beginning
                valid_data.get_range(-visualization_dataset_length, None),  # End
            ],
            "test": [
                test_data.get_range(0, visualization_dataset_length),  # Beginning
                test_data.get_range(-visualization_dataset_length, None),  # End
            ],
        }

    def general_dataloader(self, mode):
        dataset_to_use = self.dataset[mode]
        batch_sampler = SequentialDataBatchSampler(
            data_source=dataset_to_use, **self.hparams
        )
        return DataLoader(dataset_to_use, batch_sampler=batch_sampler, num_workers=4)

    @pl.data_loader
    def train_dataloader(self):
        return self.general_dataloader("train")

    @pl.data_loader
    def val_dataloader(self):
        return self.general_dataloader("valid")

    @pl.data_loader
    def test_dataloader(self):
        return self.general_dataloader("test")

    def get_dataloader(self, mode):
        if mode == "train":
            return self.train_dataloader()
        elif mode == "valid":
            return self.val_dataloader()
        elif mode == "test":
            return self.test_dataloader()
        else:
            raise NotImplementedError()

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters())
        return optim

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model to %s" % path)
        torch.save(self, path)
