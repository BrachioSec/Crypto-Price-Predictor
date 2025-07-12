# Crypto Price Predictor - Bug Fixes and Improvements

## Overview
This is a cryptocurrency price prediction application using deep learning (LSTM) with JAX/Flax framework and a Tkinter GUI.

## Bugs Fixed and Improvements Made

### 1. **Missing Function Implementations**
- **Bug**: All core functions were marked as "unchanged code" but not implemented
- **Fix**: Implemented all missing functions:
  - `fetch_data()`: Data fetching with multiple ticker format attempts
  - `compute_rsi()`: RSI technical indicator calculation
  - `compute_indicators()`: Comprehensive technical analysis indicators
  - `prepare_dataset()`: Dataset preparation with proper validation
  - `ImprovedLSTM`: Complete LSTM model architecture
  - `loss_fn()`, `train_step()`, `train_model()`: Training pipeline
  - `evaluate_model()`: Model evaluation with visualization

### 2. **Data Fetching Robustness**
- **Bug**: No error handling for different cryptocurrency ticker formats
- **Fix**: Added multiple ticker format attempts (BTC-USD, BTCUSD, BTC) with proper error handling

### 3. **GUI Thread Safety**
- **Bug**: Threading issues that could cause GUI freezing
- **Fix**: 
  - Made threads daemon threads to properly close with main program
  - Added GUI updates in the output function
  - Better error handling in threaded functions

### 4. **Model Architecture Issues**
- **Bug**: LSTM model was not properly defined for JAX/Flax
- **Fix**: 
  - Implemented proper LSTM cell scanning
  - Added dropout layers for regularization
  - Proper state initialization and management

### 5. **Data Validation**
- **Bug**: No validation for sufficient data availability
- **Fix**: 
  - Added checks for minimum data requirements
  - Feature availability validation
  - Proper error messages for insufficient data

### 6. **Visualization for Remote Environment**
- **Bug**: Matplotlib might try to display plots in headless environment
- **Fix**: 
  - Using 'Agg' backend for non-interactive plotting
  - Saving plots to files instead of displaying
  - Added plot file references in output

### 7. **Technical Indicators Implementation**
- **Bug**: Technical indicators were not implemented
- **Fix**: Added comprehensive indicators:
  - Moving averages (SMA, EMA)
  - MACD with signal and histogram
  - RSI (Relative Strength Index)
  - Bollinger Bands
  - Volume indicators
  - Price-based features and volatility

### 8. **Training Loop Improvements**
- **Bug**: Training loop was not implemented
- **Fix**: 
  - Added proper batch processing
  - Data shuffling for each epoch
  - Validation loss monitoring
  - Progress reporting every 20 epochs

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python crypto_predictor.py
```

## Usage

1. **Enter Crypto Symbol**: Input a cryptocurrency symbol (e.g., BTC, ETH, ADA)
2. **Set Prediction Horizon**: Choose how many days ahead to predict (1-30)
3. **Run Prediction**: Click the button to start training and prediction
4. **View Results**: Check the output window for detailed progress and results
5. **Generated Files**: The application will save a prediction chart as `{SYMBOL}_prediction_results.png`

## Features

- **Real-time Data Fetching**: Uses yfinance to get up-to-date cryptocurrency data
- **Technical Analysis**: Computes multiple technical indicators for better predictions
- **Deep Learning Model**: Uses LSTM neural networks with JAX/Flax for efficient training
- **Interactive GUI**: User-friendly interface with real-time progress updates
- **Visualization**: Generates charts showing actual vs predicted prices and residuals
- **Error Handling**: Robust error handling for various failure scenarios

## Model Performance Metrics

The application provides several performance metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)  
- Direction Accuracy (percentage of correct trend predictions)
- Price accuracy comparison

## Technical Details

- **Framework**: JAX/Flax for neural networks
- **Model**: Multi-layer LSTM with dropout regularization
- **Features**: 22+ technical indicators and price features
- **Training**: Adam optimizer with configurable learning rate
- **Validation**: 80/20 train/test split with proper evaluation

## Notes

- The application is designed to work in headless/remote environments
- Charts are saved as PNG files instead of being displayed
- The model trains for 100 epochs by default
- Requires at least 50+ days of historical data for reliable predictions
