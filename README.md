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

The x-Data Dimensions are : XX,25,1

The y-Data Dimensions are : XX,31,1


How Ever the main.py script is adjusted so it can use seqential data of different dimensions, since the Neural net input Layer uses x_train.shape[1] as input dimension.

The output dimesnion is set to 1, so that one point in the future is predicted instead of a sequence.




## Model

The model is based on the Transformer architecture, with some modifications to fit the problem. The architecture consists of an encoder with multiple self-attention layers, a positional encoding layer, and additional linear layers with ReLU activation.

### Model Parameters

- `d_model`: The dimension of the input and output vectors  
- `nhead`: The number of heads in the multi-head attention layer
- `num_layers`: The number of self-attention layers
- `input_size`: The dimension of the input vector
## Requirements

- Python 3.10 or higher
- NumPy
- PyTorch
- PyTorch Lightning

## Usage

To train and test the model, simply run:

```bash
python main.py
```

### Pipeline

The pipeline consists of the following steps:

- pylint (code style check) : accept only if pylint score is atleast 9/10

