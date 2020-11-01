import argparse
import os
from enum import Enum

import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Callback
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
    global args

    def objective(trial):
        class MetricsCallback(Callback):
            """PyTorch Lightning metric callback."""

            def __init__(self):
                super().__init__()
                self.metrics = []

            def on_validation_end(self, trainer, pl_module):
                self.metrics.append(trainer.callback_metrics)

        # as explained above, we'll use this callback to collect the validation accuracies
        metrics_callback = MetricsCallback()

        log_dir = "tpa_lstm_model_logs"
        log_name = "tpa_lstm_model"

        # Creating a logging object
        model_logger = TensorBoardLogger(save_dir=log_dir, name=log_name)

        # create a trainer
        trainer = pl.Trainer(
            # train_percent_check=1.0,
            # val_percent_check=1.0,
            max_epochs=3,  # epochs
            gpus=0 if torch.cuda.is_available() else None,  # #gpus
            callbacks=[metrics_callback],  # save latest accuracy
            default_root_dir=log_dir,
            logger=model_logger,
            early_stop_callback=PyTorchLightningPruningCallback(
                trial, monitor="valid_acc"
            ),
            # early stopping
        )

        # here we sample the hyper params, similar as in our old random search
        # trial_hparams = {'output_sequence_length': 30,
        #                  'num_of_batches': 64,
        #                  'batch_size': trial.suggest_int("batch_size", 8, 16),
        #                  'input_sequence_length': trial.suggest_int("input_sequence_length", 64, 128),
        #                  'hidden_size': trial.suggest_int("hidden_size", 8, 16),
        #                  'n_layers': trial.suggest_int("n_layers", 1, 2),
        #                  'num_filters': trial.suggest_int("num_filters", 8, 16),
        #                  'input_size': len(StockHistoryDataset.FEATURES),
        #                  'draw_params': {'input_kwargs': {'custom_openkwargs': {'label': None},
        #                                                   'custom_closekwargs': {'label': None},
        #                                                   'custom_rangekwargs': {'label': 'Input'}},
        #                                  'ground_truth_kwargs': {'custom_openkwargs': {'color': 'blue', 'label': None},
        #                                                          'custom_closekwargs': {'color': 'cyan', 'label': None},
        #                                                          'custom_rangekwargs': {'color': 'lightblue',
        #                                                                                 'label': 'GroundTruth'}}}}

        trial_hparams = {
            "output_sequence_length": 30,
            "num_of_batches": 64,
            "batch_size": trial.suggest_int("batch_size", 8, 64),
            "input_sequence_length": trial.suggest_int(
                "input_sequence_length", 64, 256
            ),
            "hidden_size": trial.suggest_int("hidden_size", 8, 64),
            "n_layers": trial.suggest_int("n_layers", 1, 3),
            "num_filters": trial.suggest_int("num_filters", 8, 64),
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
                    "custom_rangekwargs": {
                        "color": "lightblue",
                        "label": "GroundTruth",
                    },
                },
            },
        }

        # create model from these hyper params and train it
        dataset = load_dataset(DATASET.STOCK)
        model = create_model(dataset, trial_hparams)
        model.prepare_data()
        trainer.fit(model)

        valid_acc = metrics_callback.metrics[-1]["valid_acc"].cpu().item()

        # del trial_hparams['draw_params']
        # model_logger.experiment.add_hparams(trial_hparams, {"validation_accuracy": valid_acc})

        # return validation accuracy from latest model, as that's what we want to minimize by our hyper param search
        return valid_acc

    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    df = study.trials_dataframe()
    print(df)

    # df_records = df.to_dict(orient='records')
    #
    # # Creating a logging object
    # writer = TensorBoardLogger(
    #     save_dir='tpa_lstm_model_logs',
    #     name='model_optimization'
    # )
    #
    # for i in range(len(df_records)):
    #     df_records[i]['datetime_start'] = str(df_records[i]['datetime_start'])
    #     df_records[i]['datetime_complete'] = str(df_records[i]['datetime_complete'])
    #     df_records[i]['duration'] = str(df_records[i]['duration'])
    #     value = df_records[i].pop('value')
    #     value_dict = {'validation_accuracy': value}
    #     writer.experiment.add_hparams(df_records[i], value_dict)


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        type=DATASET,
        default=DATASET.TEMPERATURE,
        choices=list(DATASET),
    )
    args = parser.parse_args()

    main()
