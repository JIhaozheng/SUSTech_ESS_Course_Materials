This project implements electricity consumption forecasting using deep learning models, including CNN, LSTM, Transformer, and hybrid CNN-LSTM architectures.

**Dataset:** Individual household electric power consumption dataset [USI Machine Learning Repository](https://pypi.org/project/mnist-datasets/):

For more detailed information, see `report.pdf`

# Requirements

Make sure the following Python packages are installed:

`pandas numpy matplotlib seaborn scikit-learn torch pickle`

# File Structure

- `main_code.ipynb` — Main notebook for running experiments.
    
- `CNN_LSTM.py` — Script to run the CNN + LSTM hybrid model.
    
- `TimeSeriesCNN.py` — CNN model definition.
    
- `TimeSeriesLSTM.py` — LSTM model definition.
    
- `TimeSeriesTransformer.py` — Transformer model definition.
    

# Usage

1. Ensure all model definition files (`TimeSeriesCNN.py`, `TimeSeriesLSTM.py`, `TimeSeriesTransformer.py`) are in the current working directory.
    
2. Prepare the environment by importing required libraries:
    

`import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns from sklearn.preprocessing 
import MinMaxScaler 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from torch.optim.lr_scheduler import StepLR 
import pickle 
import os 
import random`

3. Run the notebook or script directly:
    

`jupyter notebook main_code.ipynb`

or

`python CNN_LSTM.py`
