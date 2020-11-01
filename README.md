# Time series prediction using sequential machine learning models

The aim of this project is to create an easy-to-use environment for developing and testing sequential machine learning models. This module uses [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) with [Tensorboard](https://www.tensorflow.org/tensorboard).

Currently, there are two example models implemented based on the projects [Deep Demand Forecast Models](https://github.com/jingw2/demand_forecast) and [Stock Prediction Models](https://github.com/huseinzol05/Stock-Prediction-Models).

**Please note, that this project is under development and not fully documented yet. If you have any problems, please create an issue.** 

## Background
Machine learning provides a flexible way to model nonlinear correlations. Its power ha been shown in many projects and more and more practical usages are coming. 

However, the standard approaches doesn't fit perfectly to sequential data. There are many approaches, two of them:
* [RNN - LSTM](https://www.simplilearn.com/tutorials/deep-learning-tutorial/rnn)
* [Attention - TPA-LSTM](https://arxiv.org/abs/1809.04206)

You can find an introduction into the topic [here](https://www.tensorflow.org/tutorials/structured_data/time_series).

## The framework
The project uses [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and provides base for sequential models.

### DataSampler
You can find a base class `SequentialDataBatchSampler` for sampling subsequences from a longer sequence for training. 

### DataSet
You can find a base class `SequentialDataset` for sequential datasets


## References
* [Deep Demand Forecast Models](https://github.com/jingw2/demand_forecast)
* [Stock Prediction Models](https://github.com/huseinzol05/Stock-Prediction-Models)
* [TPA-LSTM](https://github.com/gantheory/TPA-LSTM)
