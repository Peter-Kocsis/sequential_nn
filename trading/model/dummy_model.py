import torch
from torch import nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader, SubsetRandomSampler

from trading.dataloader.sequential_dataset import SequentialDataset
from trading.dataloader.stock_dataset import StockHistoryDataset
import pandas as pd


class DummyModel(pl.LightningModule):
    def __init__(self, network: nn.Module, dataset: SequentialDataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.network = network

    def forward(self, input):
        return self.network(input)

    def plot(
        self, input: StockHistoryDataset, ground_truth: StockHistoryDataset = None
    ):

        prediction = self.network(input)

        input_length = len(input)
        prediction_length = len(prediction)

        plt.figure(1, figsize=((input_length + prediction_length) / 300, 10), dpi=200)
        StockHistoryDataset.draw(
            input.data,
            custom_openkwargs={"label": None},
            custom_closekwargs={"label": None},
            custom_rangekwargs={"label": "Input"},
        )

        last_x = input[-1]["Date"]
        ymin = min(prediction)
        ymax = max(prediction)
        plt.vlines(
            last_x, color="blue", linestyles="dashed", ymin=ymin, ymax=ymax, linewidth=2
        )

        prediction_x = pd.Series(
            pd.date_range(last_x, periods=prediction_length, freq="D")
        )
        plt.plot(prediction_x, prediction, color="red", label="Prediction")

        if ground_truth is not None:
            StockHistoryDataset.draw(
                ground_truth.data,
                custom_openkwargs={"color": "blue", "label": None},
                custom_closekwargs={"color": "cyan", "label": None},
                custom_rangekwargs={"color": "lightblue", "label": "GroundTruth"},
            )

        plt.title(f"Model prediction - {self.network}")
        plt.legend()
        plt.show()

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # load X, y to device!
        images, targets = images.to(self.device), targets.to(self.device)

        # forward pass
        outs = self.forward(images)

        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
        loss = loss_func(outs, targets)

        # accuracy
        _, preds = torch.max(outs, 1)
        accuracy = torch.mean((preds == targets).float()[targets >= 0])

        return loss, accuracy

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + "_loss"] for x in outputs]).mean()
        avg_accuracy = torch.stack([x[mode + "_accuracy"] for x in outputs]).mean()

        return avg_loss, avg_accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {"loss": loss, "accuracy": accuracy}
        return {"loss": loss, "accuracy": accuracy, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "val")
        return {"val_loss": loss, "val_accuracy": accuracy}

    def test_step(self, batch, batch_idx):
        loss, accuracy = self.general_step(batch, batch_idx, "test")
        return {"test_loss": loss, "test_accuracy": accuracy}

    def validation_end(self, outputs):
        avg_loss, avg_accuracy = self.general_end(outputs, "val")
        tensorboard_logs = {"val_loss": avg_loss, "val_accuracy": avg_accuracy}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def prepare_data(self):
        # create dataset
        download_url = (
            "http://filecremers3.informatik.tu-muenchen.de/~dl4cv/segmentation_data.zip"
        )
        i2dl_exercises_path = os.path.dirname(os.path.abspath(os.getcwd()))
        data_root = os.path.join(i2dl_exercises_path, "datasets", "segmentation")

        train_data = SegmentationData(
            image_paths_file=f"{data_root}/segmentation_data/train.txt"
        )
        val_data = SegmentationData(
            image_paths_file=f"{data_root}/segmentation_data/val.txt"
        )
        test_data = SegmentationData(
            image_paths_file=f"{data_root}/segmentation_data/test.txt"
        )

        # assign to use in dataloaders
        self.dataset = {}
        self.dataset["train"], self.dataset["val"], self.dataset["test"] = (
            train_data,
            val_data,
            test_data,
        )

    @pl.data_loader
    def train_dataloader(self):
        sampler = SubsetRandomSampler()
        return DataLoader(
            self.dataset["train"], shuffle=True, batch_size=self.batch_size
        )

    # @pl.data_loader
    # def val_dataloader(self):
    #     return DataLoader(self.dataset["val"], batch_size=self.batch_size)
    #
    # @pl.data_loader
    # def test_dataloader(self):
    #     return DataLoader(self.dataset["test"], batch_size=self.batch_size)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), self.learning_rate)
        return optim

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print("Saving model... %s" % path)
        torch.save(self.state_dict(), path)
        tmp_model = SegmentationNN(hparams=self.hparams)
        tmp_model.load_state_dict(torch.load(path))
        tmp_model.eval()
        torch.save(tmp_model, path)
