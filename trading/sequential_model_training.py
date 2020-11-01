import argparse
import os
from enum import Enum

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from trading.dataloader.sequential_dataset import SequentialDataset
from trading.dataloader.stock_dataset import (
    StockParams,
    StockHistoryDataset,
    StockHistoryUpdateMode,
)
from trading.dataloader.temperature_dataset import TemperatureDataset
from trading.model.tpa_lstm_model import TPA_LSTM_Model


class DATASET(Enum):
    TEMPERATURE = "temperature"
    STOCK = "stock"


hparams_temperature = {
    "output_sequence_length": 30,
    "num_of_batches": 50,
    "batch_size": 10,
    "input_sequence_length": 128,
    "hidden_size": 24,
    "n_layers": 1,
    "num_filters": 32,
    "input_size": len(TemperatureDataset.FEATURES),
    "draw_params": {
        "input_kwargs": {"label": "Input"},
        "ground_truth_kwargs": {"color": "lightblue", "label": "GroundTruth"},
    },
}
hparams_stock = {
    "output_sequence_length": 30,
    "num_of_batches": 64,
    "batch_size": 20,
    "input_sequence_length": 128,
    "hidden_size": 32,
    "n_layers": 1,
    "num_filters": 16,
    "input_size": len(StockHistoryDataset.FEATURES),
    "draw_params": {
        "input_kwargs": {
            "custom_openkwargs": {"label": None},
            "custom_closekwargs": {"label": None},
            "custom_rangekwargs": {"label": "Input"},
        },
        "ground_truth_kwargs": {
            "custom_openkwargs": {"color": "blue", "label": None},
            "custom_closekwargs": {"color": "cyan", "label": None},
            "custom_rangekwargs": {"color": "lightblue", "label": "GroundTruth"},
        },
    },
}


def load_dataset(dataset_to_load: DATASET) -> SequentialDataset:
    # Create dataset
    root_folder = os.path.abspath(os.getcwd())
    data_folder = os.path.join(root_folder, "data")

    if dataset_to_load == DATASET.TEMPERATURE:
        dataset = TemperatureDataset(root_path=data_folder)
    elif dataset_to_load == DATASET.STOCK:
        stock_symbol = "BAS.DE"
        period = "max"
        stock_params = StockParams(symbol=stock_symbol, period=period)

        root_folder = os.path.abspath(os.getcwd())
        data_folder = os.path.join(root_folder, "data")

        dataset = StockHistoryDataset(
            root_path=data_folder,
            stock_params=stock_params,
            update_mode=StockHistoryUpdateMode.KEEP,
        )
    else:
        raise NotImplementedError()

    return dataset


def get_config(dataset_to_load: DATASET) -> dict:
    if dataset_to_load == DATASET.TEMPERATURE:
        return hparams_temperature
    elif dataset_to_load == DATASET.STOCK:
        return hparams_stock
    else:
        raise NotImplementedError()


def create_model(dataset, hparams) -> TPA_LSTM_Model:
    model = TPA_LSTM_Model(hparams=hparams, sequential_dataset=dataset)

    return model


def fit_model(model: TPA_LSTM_Model, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)

    epochs = 50

    # Creating a logging object
    segmentation_nn_logger = TensorBoardLogger(
        save_dir=log_dir, name="temperature_model"
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=False,
        monitor="loss",
        mode="min",
        prefix="best",
    )

    trainer = pl.Trainer(
        checkpoint_callback=checkpoint_callback,
        default_root_dir="../models/",
        max_epochs=epochs,
        logger=segmentation_nn_logger,
        gpus=-1,
    )
    trainer.fit(model)
    trainer.test()

    model.eval()
    model.freeze()


def load_model(chechpoint_path: str) -> TPA_LSTM_Model:
    model = TPA_LSTM_Model.load_from_checkpoint(checkpoint_path=chechpoint_path)
    return model


def plot_prediction(
    input: SequentialDataset, ground_truth: SequentialDataset, model: TPA_LSTM_Model
):
    model.plot(input, ground_truth)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=DATASET,
        default=DATASET.TEMPERATURE,
        choices=list(DATASET),
    )
    arguments = parser.parse_args()
    dataset_to_load = arguments.dataset
    hparams = get_config(dataset_to_load)

    dataset = load_dataset(dataset_to_load)
    model = create_model(dataset, hparams)
    fit_model(model, "tpa_lstm_model_logs")

    # model = load_model("./models/temperature_model/version_58/checkpoints/bestlast.ckpt")

    input_data = dataset.get_range(
        -hparams["input_sequence_length"] - hparams["output_sequence_length"],
        -hparams["output_sequence_length"],
    )
    ground_truth = dataset.get_range(-hparams["output_sequence_length"], None)
    plot_prediction(input_data, ground_truth, model)


if __name__ == "__main__":
    main()
