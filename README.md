# Signal Power Prediction with Transformer

This repository contains the implementation of a Transformer-based model for predicting carrier signal power values. The model uses a custom Transformer architecture and processes the input time series data to make predictions. The main goal of this project is to predict the signal power values in a noisy environment accurately.



## Models

To compare the comparisson of the transformer model with other architectures there are 3 other models implemented:

- LSTM  
- Conv 1D
- MLP

Thus there are 4 models in total. The models are implemented in PyTorch and are located at models

To have a baseline comparisson there is also a wiener filter implemented. The wiener filter is located at models/wiener_filter.py

## Requirements

- Python 3.10 or higher
- NumPy
- PyTorch
- PyTorch Lightning

## Usage

To train and test all architectures run :

```bash
python train_allmodels.py
```

If you just want to train the Transformermodel run :

```bash
python train_transformer.py
```

## Problems 

If your System has a GPU but doesn't support distributed Training try:

```python
trainer = pl.Trainer(max_epochs=MAX_EPOCHS,
                         callbacks=[early_stop_callback],
                         logger=logger,
                         accelerator="gpu",
                         devices=1)
```
instead of 

```python
trainer = pl.Trainer(max_epochs=MAX_EPOCHS, callbacks=[early_stop_callback], logger=logger)                     
```