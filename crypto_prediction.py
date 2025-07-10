import requests
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for remote environment
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Data Fetching ---
def fetch_data(symbol, days=730):  # Default 2 years of data
    """Fetch recent cryptocurrency data"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    ticker = symbol.upper() + "-USD"
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")
    
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # Handle MultiIndex columns by flattening them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"Downloaded {len(df)} days of data")
    return df

# --- Technical Indicators ---
def compute_rsi(series, period=14):
    delta = series.diff().dropna()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_indicators(df):
    """Compute technical indicators and normalize them"""
    close = df['Close']
    
    # Price-based indicators
    df['SMA20'] = close.rolling(window=20).mean()
    df['EMA20'] = close.ewm(span=20).mean()
    df['RSI14'] = compute_rsi(close)
    
    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACDsig'] = df['MACD'].ewm(span=9).mean()
    
    # Bollinger Bands
    std20 = close.rolling(window=20).std()
    df['BB_up'] = df['SMA20'] + 2 * std20
    df['BB_dn'] = df['SMA20'] - 2 * std20
    
    # Volume-based indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Price change indicators
    df['Price_Change'] = close.pct_change()
    df['Volatility'] = close.rolling(window=20).std() / close.rolling(window=20).mean()
    
    return df.dropna()

# --- Dataset Preparation with Normalization ---
def prepare_dataset(df, target_col='Close', window=30, horizon=1):
    """Prepare dataset with proper normalization"""
    print(f"Preparing dataset with window={window}, horizon={horizon}")
    
    # Create a copy and reset index
    df_copy = df.copy().reset_index(drop=True)
    
    # Select features (exclude target and date columns)
    feature_cols = [col for col in df_copy.select_dtypes(include=[np.number]).columns 
                   if col != target_col]
    
    # Separate features and target
    features = df_copy[feature_cols]
    target = df_copy[target_col].values
    
    # Normalize features
    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)
    
    # Normalize target (prices)
    target_scaler = MinMaxScaler()
    target_scaled = target_scaler.fit_transform(target.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, Y = [], []
    for i in range(len(df_copy) - window - horizon + 1):
        X.append(features_scaled[i:i+window])
        Y.append(target_scaled[i+window+horizon-1])
    
    X = jnp.array(X, dtype=jnp.float32)
    Y = jnp.array(Y, dtype=jnp.float32)
    
    print(f"Created {len(X)} sequences with {X.shape[2]} features")
    
    return X, Y, feature_scaler, target_scaler

# --- Improved Model Definition ---
class ImprovedLSTM(nn.Module):
    hidden_size: int = 64

    def setup(self):
        self.lstm_cell1 = nn.OptimizedLSTMCell(features=self.hidden_size)
        self.lstm_cell2 = nn.OptimizedLSTMCell(features=self.hidden_size//2)
        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(1)

    def __call__(self, x, training=True):
        batch_size, seq_len, _ = x.shape
        
        # First LSTM layer
        carry1 = self.lstm_cell1.initialize_carry(jax.random.PRNGKey(0), (batch_size,))
        outputs1 = []
        for i in range(seq_len):
            carry1, output1 = self.lstm_cell1(carry1, x[:, i, :])
            outputs1.append(output1)
        
        # Second LSTM layer
        carry2 = self.lstm_cell2.initialize_carry(jax.random.PRNGKey(1), (batch_size,))
        outputs2 = []
        for i in range(seq_len):
            carry2, output2 = self.lstm_cell2(carry2, outputs1[i])
            outputs2.append(output2)
        
        # Use last output
        last_output = outputs2[-1]
        
        # Apply dense layers with optional dropout during training
        x = nn.relu(self.dense1(last_output))
        if training:
            # Simple noise injection instead of complex dropout
            noise = jax.random.normal(jax.random.PRNGKey(42), x.shape) * 0.1
            x = x + noise
        
        output = self.dense2(x)
        return output.squeeze(-1)

# --- Training Utilities ---
def loss_fn(params, apply_fn, x, y, training=True):
    preds = apply_fn({'params': params}, x, training=training)
    return jnp.mean((preds - y) ** 2)

def train_step(state, apply_fn, x, y):
    def loss_wrapper(params):
        return loss_fn(params, apply_fn, x, y, training=True)
    
    grads = jax.grad(loss_wrapper)(state.params)
    return state.apply_gradients(grads=grads)

# --- Training Loop ---
def train_model(model, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001):
    """Train the model with better parameters"""
    rng = jax.random.PRNGKey(42)
    
    # Create training state
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, X_train[:1], training=True)['params'],
        tx=optax.adam(lr)
    )
    
    best_test_loss = float('inf')
    patience = 20
    no_improve = 0
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training step
        state = train_step(state, model.apply, X_train, Y_train)
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_loss = loss_fn(state.params, model.apply, X_train, Y_train, training=False)
            test_loss = loss_fn(state.params, model.apply, X_test, Y_test, training=False)
            
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.6f}, Test Loss={test_loss:.6f}")
            
            # Early stopping
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                no_improve = 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return state

# --- Evaluation and Prediction ---
def evaluate_model(state, model, X_test, Y_test, target_scaler, symbol, horizon):
    """Evaluate model and create realistic predictions"""
    # Get predictions (normalized)
    preds_norm = model.apply({'params': state.params}, X_test, training=False)
    
    # Denormalize predictions and targets
    preds_real = target_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
    y_test_real = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((preds_real - y_test_real) ** 2)
    mae = np.mean(np.abs(preds_real - y_test_real))
    mape = np.mean(np.abs((y_test_real - preds_real) / y_test_real)) * 100
    
    print(f"\nModel Evaluation:")
    print(f"MSE: ${mse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    # Plot last 100 points for clarity
    plot_points = min(100, len(y_test_real))
    x_axis = range(plot_points)
    
    plt.subplot(2, 1, 1)
    plt.plot(x_axis, y_test_real[-plot_points:], 'b-', label="Actual Price", linewidth=2)
    plt.plot(x_axis, preds_real[-plot_points:], 'r--', label="Predicted Price", linewidth=2)
    plt.title(f"{symbol} Price Prediction ({horizon} day horizon) - Last {plot_points} predictions")
    plt.xlabel("Time Steps")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error
    plt.subplot(2, 1, 2)
    error = y_test_real[-plot_points:] - preds_real[-plot_points:]
    plt.plot(x_axis, error, 'g-', alpha=0.7)
    plt.title("Prediction Error")
    plt.xlabel("Time Steps")
    plt.ylabel("Error (USD)")
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{symbol}_realistic_prediction.png', dpi=150, bbox_inches='tight')
    print(f"Realistic prediction plot saved as {symbol}_realistic_prediction.png")
    
    return preds_real, y_test_real

# --- Main Execution ---
def main():
    symbol = input("Enter crypto symbol (e.g., BTC): ").upper()
    horizon = int(input("Enter prediction horizon (days, recommended 1-7): "))
    
    # Validate horizon
    if horizon > 30:
        print("Warning: Large horizons (>30 days) may not be realistic for crypto prediction")
    
    # Fetch and prepare data
    df = fetch_data(symbol, days=730)  # 2 years of data
    df = compute_indicators(df)
    
    # Prepare dataset with normalization
    X, Y, feature_scaler, target_scaler = prepare_dataset(df, horizon=horizon, window=30)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and train model
    model = ImprovedLSTM()
    state = train_model(model, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001)
    
    # Evaluate model
    preds_real, y_test_real = evaluate_model(state, model, X_test, Y_test, target_scaler, symbol, horizon)
    
    print(f"\nLatest actual price: ${y_test_real[-1]:.2f}")
    print(f"Latest predicted price: ${preds_real[-1]:.2f}")
    print(f"Price difference: ${abs(y_test_real[-1] - preds_real[-1]):.2f}")

if __name__ == "__main__":
    main()