# Signal Power Prediction with Transformer

This repository contains the implementation of a Transformer-based model for predicting carrier signal power values. The model uses a custom Transformer architecture and processes the input time series data to make predictions. The main goal of this project is to predict the signal power values in a noisy environment accurately.

## Dataset

The dataset consists of signal power values collected from different carriers. It is divided into three parts: training, validation, and test sets. The data is stored as NumPy arrays in the following format:

- `x_train_snrXX.npy`: Training input data with XX dB SNR
- `y_train_snrXX.npy`: Training output/target data with XX dB SNR
- `x_val_snrXX.npy`: Validation input data with XX dB SNR
- `y_val_snrXX.npy`: Validation output/target data with XX dB SNR
- `x_test_snrXX.npy`: Test input data with XX dB SNR
- `y_test_snrXX.npy`: Test output/target data with XX dB SNR

## Model

The model is based on the Transformer architecture, with some modifications to fit the problem. The architecture consists of an encoder with multiple self-attention layers, a positional encoding layer, and additional linear layers with ReLU activation.

### Model Parameters

- `num_layers`: Number of layers in the Transformer Encoder
- `d_model`: Dimension of the model
- `num_heads`: Number of attention heads
- `dff`: Dimension of the feedforward network inside the Transformer layers
- `dropout_rate`: Dropout rate for regularization

## Requirements

- Python 3.7 or higher
- NumPy
- PyTorch

## Usage

To train and test the model, simply run:

```bash
python main.py

