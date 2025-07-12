import threading
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText

import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for remote environment
import matplotlib.pyplot as plt
import seaborn as sns
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set better plotting style
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

# --- Data Fetching with Better Error Handling ---

def fetch_data(symbol, days=730):
    """Fetch cryptocurrency data with error handling"""
    try:
        # Try different ticker formats
        possible_tickers = [
            f"{symbol}-USD",
            f"{symbol}USD",
            symbol
        ]
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for ticker in possible_tickers:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty and len(data) > 50:  # Ensure we have enough data
                    # Clean the data
                    data = data.dropna()
                    if len(data) > 50:
                        return data, ticker
            except Exception as e:
                continue
        
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# --- Technical Indicators ---
def compute_rsi(series, period=14):
    """Compute Relative Strength Index"""
    if len(series) < period:
        return pd.Series([np.nan] * len(series), index=series.index)
    
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_indicators(df):
    """Compute technical indicators"""
    df = df.copy()
    
    # Moving averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential moving averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    df['RSI'] = compute_rsi(df['Close'])
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=10).mean()
    
    # Price-based features
    df['Price_Change'] = df['Close'].pct_change()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=10).std()
    
    return df.dropna()

# --- Dataset Preparation with Better Validation ---
def prepare_dataset(df, target_col='Close', window=30, horizon=1):
    """Prepare dataset for training"""
    # Select features
    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Histogram',
        'RSI', 'BB_Middle', 'BB_Upper', 'BB_Lower',
        'Volume_SMA', 'Price_Change', 'High_Low_Ratio',
        'Price_Position', 'Volatility'
    ]
    
    # Ensure all features exist
    available_features = [col for col in feature_cols if col in df.columns]
    
    if len(available_features) < 5:
        raise ValueError(f"Not enough features available. Found: {available_features}")
    
    # Scale features
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(df[available_features].values)
    target_scaled = target_scaler.fit_transform(df[[target_col]].values)
    
    # Create sequences
    X, Y = [], []
    for i in range(window, len(features_scaled) - horizon + 1):
        X.append(features_scaled[i-window:i])
        Y.append(target_scaled[i+horizon-1, 0])
    
    if len(X) == 0:
        raise ValueError("Not enough data to create sequences")
    
    return np.array(X), np.array(Y), feature_scaler, target_scaler

# --- Improved Model Definition ---
class ImprovedLSTM(nn.Module):
    """Improved LSTM model with dropout and multiple layers"""
    
    def setup(self):
        self.lstm1 = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )()
        
        self.lstm2 = nn.scan(
            nn.OptimizedLSTMCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1
        )()
        
        self.dropout = nn.Dropout(rate=0.2)
        self.dense1 = nn.Dense(50)
        self.dense2 = nn.Dense(25)
        self.output_layer = nn.Dense(1)
    
    def __call__(self, x, training=False):
        batch_size, seq_len, features = x.shape
        
        # Initialize LSTM states
        carry1 = self.lstm1.initialize_carry(jax.random.PRNGKey(0), (batch_size, 64))
        carry2 = self.lstm2.initialize_carry(jax.random.PRNGKey(1), (batch_size, 32))
        
        # First LSTM layer
        carry1, x = self.lstm1(carry1, x)
        x = self.dropout(x, deterministic=not training)
        
        # Second LSTM layer
        carry2, x = self.lstm2(carry2, x)
        x = self.dropout(x, deterministic=not training)
        
        # Take the last output
        x = x[:, -1, :]
        
        # Dense layers
        x = self.dense1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training)
        
        x = self.dense2(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training)
        
        x = self.output_layer(x)
        
        return x.squeeze(-1)

# --- Training Utilities ---
def loss_fn(params, apply_fn, x, y, training=True):
    """Mean squared error loss function"""
    pred = apply_fn(params, x, training=training)
    return jnp.mean((pred - y) ** 2)

def train_step(state, apply_fn, x, y):
    """Single training step"""
    def loss_fn_wrapper(params):
        return loss_fn(params, apply_fn, x, y, training=True)
    
    loss, grads = jax.value_and_grad(loss_fn_wrapper)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# --- Training Loop ---
def train_model(model, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001):
    """Train the model"""
    # Initialize model
    rng = jax.random.PRNGKey(42)
    sample_input = jnp.ones((1, X_train.shape[1], X_train.shape[2]))
    params = model.init(rng, sample_input, training=False)
    
    # Create optimizer
    optimizer = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )
    
    # Convert to JAX arrays
    X_train_jax = jnp.array(X_train)
    Y_train_jax = jnp.array(Y_train)
    X_test_jax = jnp.array(X_test)
    Y_test_jax = jnp.array(Y_test)
    
    batch_size = min(32, len(X_train))
    
    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = X_train_jax[batch_indices]
            batch_y = Y_train_jax[batch_indices]
            
            state, loss = train_step(state, model.apply, batch_x, batch_y)
            epoch_loss += loss
            num_batches += 1
        
        if epoch % 20 == 0:
            avg_loss = epoch_loss / num_batches
            # Validation loss
            val_loss = loss_fn(state.params, model.apply, X_test_jax, Y_test_jax, training=False)
            print(f"Epoch {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    return state

# --- Enhanced Evaluation and Visualization ---
def evaluate_model(state, model, X_test, Y_test, target_scaler, symbol, horizon, ticker_used):
    """Evaluate model and create visualizations"""
    # Make predictions
    predictions = model.apply(state.params, jnp.array(X_test), training=False)
    
    # Inverse transform
    predictions_real = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test_real = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((predictions_real - y_test_real) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_real - y_test_real))
    
    # Calculate accuracy based on direction
    direction_pred = np.diff(predictions_real) > 0
    direction_actual = np.diff(y_test_real) > 0
    direction_accuracy = np.mean(direction_pred == direction_actual) * 100
    
    print(f"\n📊 Model Performance:")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   Direction Accuracy: {direction_accuracy:.1f}%")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot predictions vs actual
    plt.subplot(2, 1, 1)
    plt.plot(y_test_real[-100:], label='Actual', color='blue', linewidth=2)
    plt.plot(predictions_real[-100:], label='Predicted', color='red', linewidth=2, alpha=0.7)
    plt.title(f'{symbol} Price Prediction - Last 100 Test Points')
    plt.xlabel('Time Steps')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot residuals
    plt.subplot(2, 1, 2)
    residuals = y_test_real - predictions_real
    plt.plot(residuals[-100:], color='green', alpha=0.7)
    plt.title('Prediction Residuals')
    plt.xlabel('Time Steps')
    plt.ylabel('Residual ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{symbol}_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure since we're using Agg backend
    
    return predictions_real, y_test_real

# --- GUI Application ---
class CryptoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Crypto Price Predictor")
        self.geometry("700x500")

        ttk.Label(self, text="Crypto Symbol (BTC, ETH, ADA):").pack(pady=(20,5))
        self.symbol_var = tk.StringVar()
        self.symbol_entry = ttk.Entry(self, textvariable=self.symbol_var, width=20)
        self.symbol_entry.pack()

        ttk.Label(self, text="Prediction Horizon (days):").pack(pady=(20,5))
        self.horizon_var = tk.IntVar(value=1)
        self.horizon_spin = ttk.Spinbox(self, from_=1, to=30, textvariable=self.horizon_var, width=5)
        self.horizon_spin.pack()

        self.run_btn = ttk.Button(self, text="Run Prediction", command=self.run_prediction_thread)
        self.run_btn.pack(pady=20)

        self.output_text = ScrolledText(self, height=15, width=80, state='disabled', font=("Consolas", 10))
        self.output_text.pack(padx=10, pady=10)

    def append_output(self, text):
        self.output_text.configure(state='normal')
        self.output_text.insert(tk.END, text + "\n")
        self.output_text.see(tk.END)
        self.output_text.configure(state='disabled')
        self.update()  # Update GUI

    def run_prediction_thread(self):
        t = threading.Thread(target=self.run_prediction)
        t.daemon = True  # Make thread daemon so it closes when main program closes
        t.start()

    def run_prediction(self):
        symbol = self.symbol_var.get().strip().upper()
        horizon = self.horizon_var.get()

        if not symbol:
            messagebox.showerror("Input Error", "Please enter a crypto symbol.")
            return

        if not (1 <= horizon <= 30):
            messagebox.showerror("Input Error", "Horizon must be between 1 and 30.")
            return

        self.run_btn.config(state='disabled')
        self.output_text.configure(state='normal')
        self.output_text.delete('1.0', tk.END)
        self.output_text.configure(state='disabled')

        try:
            self.append_output(f"🚀 Fetching data for {symbol} ...")
            result = fetch_data(symbol, days=730)
            if result is None:
                raise ValueError("Data could not be fetched for the given symbol.")
            df, ticker_used = result

            self.append_output(f"✅ Data fetched successfully for {ticker_used}")
            self.append_output(f"   Total records: {len(df)}")
            self.append_output(f"   Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

            self.append_output("⚙️ Computing indicators...")
            df = compute_indicators(df)
            self.append_output(f"   Indicators computed. Final dataset size: {len(df)}")

            self.append_output(f"🛠 Preparing dataset with horizon={horizon} ...")
            X, Y, feature_scaler, target_scaler = prepare_dataset(df, horizon=horizon, window=30)

            split_idx = int(0.8 * len(X))
            X_train, Y_train = X[:split_idx], Y[:split_idx]
            X_test, Y_test = X[split_idx:], Y[split_idx:]

            self.append_output(f"📊 Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
            self.append_output(f"   Feature dimensions: {X_train.shape}")

            self.append_output("🧠 Initializing and training model...")
            model = ImprovedLSTM()
            state = train_model(model, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001)

            self.append_output("🔍 Evaluating model ...")
            preds_real, y_test_real = evaluate_model(state, model, X_test, Y_test, target_scaler, symbol, horizon, ticker_used)

            price_diff = abs(y_test_real[-1] - preds_real[-1])
            accuracy = (1 - price_diff / y_test_real[-1]) * 100

            self.append_output(f"\n🎯 Final Results:")
            self.append_output(f"   Latest actual price: ${y_test_real[-1]:,.2f}")
            self.append_output(f"   Latest predicted price: ${preds_real[-1]:,.2f}")
            self.append_output(f"   Price difference: ${price_diff:,.2f}")
            self.append_output(f"   Accuracy: {accuracy:.1f}%")
            self.append_output(f"   Chart saved as: {symbol}_prediction_results.png")

            messagebox.showinfo("Prediction Complete", "Prediction and evaluation finished. Check the output and saved chart.")

        except Exception as e:
            self.append_output(f"\n❌ Error: {str(e)}")
            messagebox.showerror("Error", str(e))
        finally:
            self.run_btn.config(state='normal')

if __name__ == "__main__":
    app = CryptoApp()
    app.mainloop()