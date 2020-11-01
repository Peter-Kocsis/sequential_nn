import os
from datetime import date

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from trading.dataloader.sequential_dataset import SequentialDataset


class TemperatureDataset(SequentialDataset):

    HEADERS = ["Date", "Hour", "MT_200", "Year", "Month", "Day", "DayOfWeek"]
    FEATURES = ["MT_200", "Year", "Month", "Day", "DayOfWeek", "Hour"]
    OUTPUTS = ["MT_200"]

    def __init__(self, *args, root_path: str, **kwargs):
        super().__init__(*args, root_path=root_path, **kwargs)
        self.name = "LD_MT200_hour"
        self.load()

    def load(self):
        print(f"Loading the dataset - {self.name}")
        dataset_path = self.get_dataset_path()
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"The dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path)
        data["Date"] = pd.to_datetime(data["Date"]) + data["Hour"].astype(
            "timedelta64[h]"
        )
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data["DayOfWeek"] = data["Date"].dt.dayofweek

        data = data.loc[
            (data["Date"] >= date(2014, 1, 1)) & (data["Date"] <= date(2014, 3, 1))
        ]

        self.data_source = data

    def get_dataset_path(self):
        return os.path.join(self.root_path, self.name + ".csv")

    @staticmethod
    def draw(data_source, ax=None, custom_draw_kwargs=None):

        if ax is None:
            ax = plt
        value_kwargs = {"color": "black", "label": "Open"}

        if custom_draw_kwargs is not None:
            value_kwargs.update(custom_draw_kwargs)

        x = data_source["Date"]
        ax.plot(x, data_source["MT_200"], **value_kwargs)
