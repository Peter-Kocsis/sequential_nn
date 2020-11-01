"""Data utility functions."""
import copy
import os
from enum import Enum

import torch
from torch.utils.data import Dataset
import yfinance as yf
import pandas as pd

import matplotlib.pyplot as plt

from trading.dataloader.sequential_dataset import SequentialDataset


class StockHistoryUpdateMode(Enum):
    KEEP = "keep"
    UPDATE = "update"
    AUTOMATIC = "automatic"


class StockParams:
    def __init__(self, symbol, period):
        self.symbol = symbol
        self.period = period

    def __repr__(self):
        return f"{self.symbol}_{self.period}"

    def __str__(self):
        return f"Stock {self.symbol}, period {self.period}"


class StockHistoryDataset(SequentialDataset):

    HEADERS = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Year",
        "Month",
        "Day",
        "DayOfWeek",
    ]
    FEATURES = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
        "Month",
        "Day",
        "DayOfWeek",
    ]
    OUTPUTS = ["Open"]

    def __init__(
        self,
        *args,
        root_path: str,
        stock_params: StockParams,
        update_mode=StockHistoryUpdateMode.KEEP,
        **kwargs,
    ):
        super().__init__(*args, root_path=root_path, **kwargs)
        self.stock_params = stock_params

        self.name = str(self.stock_params)
        self.prepare(update_mode)

    def prepare(self, update_mode: StockHistoryUpdateMode):
        print(f"Preparing the dataset - {self.stock_params}")
        if update_mode == StockHistoryUpdateMode.KEEP:
            pass
        elif update_mode == StockHistoryUpdateMode.UPDATE:
            self.update()
        self.load()
        print(f"Dataset prepared - {self.stock_params}")

    def update(self):
        stock_symbol = self.stock_params.symbol
        stock_period = self.stock_params.period

        stock_ticker = yf.Ticker(stock_symbol)
        print(f"Updating the dataset - {self.stock_params}")
        stock_data = stock_ticker.history(period=stock_period, auto_adjust=False)

        os.makedirs(os.path.dirname(self.root_path), exist_ok=True)
        stock_data.to_csv(self.get_dataset_path())

    def load(self):
        print(f"Loading the dataset - {self.stock_params}")
        dataset_path = self.get_dataset_path()
        if not os.path.isfile(dataset_path):
            raise FileNotFoundError(f"The dataset not found: {dataset_path}")

        data = pd.read_csv(dataset_path, parse_dates=["Date"])
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data["DayOfWeek"] = data["Date"].dt.dayofweek

        self.data_source = data

    def get_dataset_path(self):
        return os.path.join(self.root_path, repr(self.stock_params) + ".csv")

    @staticmethod
    def draw(data, ax=None, custom_draw_kwargs=None):

        if ax is None:
            ax = plt

        if custom_draw_kwargs is None:
            custom_draw_kwargs = {}

        rangekwargs = {"color": "whitesmoke", "label": "Price range"}
        closekwargs = {"color": "grey", "label": "Close"}
        openkwargs = {"color": "black", "label": "Open"}

        if custom_draw_kwargs.get("custom_rangekwargs", None) is not None:
            rangekwargs.update(custom_draw_kwargs["custom_rangekwargs"])
        if custom_draw_kwargs.get("custom_closekwargs", None) is not None:
            closekwargs.update(custom_draw_kwargs["custom_closekwargs"])
        if custom_draw_kwargs.get("custom_openkwargs", None) is not None:
            openkwargs.update(custom_draw_kwargs["custom_openkwargs"])

        x = data["Date"]
        price_average = (data["High"] + data["Low"]) / 2

        ax.plot(x, data["Open"], **openkwargs)
        ax.plot(x, data["Close"], **closekwargs)

        ax.errorbar(x, price_average, yerr=data["High"] - price_average, **rangekwargs)
