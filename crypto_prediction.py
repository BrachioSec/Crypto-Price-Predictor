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

# --- Data Fetching ---
def fetch_data(symbol, start="2022-01-01", end="2025-01-01"):
    ticker = symbol.upper() + "-USD"
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    
    # Handle MultiIndex columns by flattening them
    if isinstance(df.columns, pd.MultiIndex):
        # For single ticker, just take the price columns and flatten
        df.columns = df.columns.get_level_values(0)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

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
    close = df['Close']
    df['SMA20'] = close.rolling(window=20).mean()
    df['EMA20'] = close.ewm(span=20).mean()
    df['RSI14'] = compute_rsi(close)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACDsig'] = df['MACD'].ewm(span=9).mean()
    # Fix the standard deviation calculation
    std20 = close.rolling(window=20).std()
    df['BB_up'] = df['SMA20'] + 2 * std20
    df['BB_dn'] = df['SMA20'] - 2 * std20
    return df.dropna()

# --- Dataset Preparation ---
def prepare_dataset(df, target_col='Close', window=60, horizon=1):
    # Create a copy and reset index to avoid multi-index issues
    df_copy = df.copy().reset_index(drop=True)
    # Ensure we only use numeric columns and drop the target column
    features = df_copy.drop(columns=[target_col]).select_dtypes(include=[np.number])
    target = df_copy[target_col].values
    X, Y = [], []
    for i in range(len(df_copy) - window - horizon + 1):
        X.append(features.iloc[i:i+window].values)
        Y.append(target[i+window+horizon-1])
    return jnp.array(X, dtype=jnp.float32), jnp.array(Y, dtype=jnp.float32)

# --- Model Definition ---
class LSTM(nn.Module):
    hidden_size: int = 128

    def setup(self):
        self.lstm_cell = nn.OptimizedLSTMCell(features=self.hidden_size)
        self.dense = nn.Dense(1)

    def __call__(self, x):
        # x shape: [batch, seq_len, features]
        batch_size, seq_len, _ = x.shape
        
        # Initialize carry state
        carry = self.lstm_cell.initialize_carry(
            jax.random.PRNGKey(0), 
            (batch_size,)
        )
        
        # Process each time step
        outputs = []
        for i in range(seq_len):
            carry, output = self.lstm_cell(carry, x[:, i, :])
            outputs.append(output)
        
        # Use the last output
        last_output = outputs[-1]
        
        # Apply dense layer and return scalar prediction
        return self.dense(last_output).squeeze(-1)

# --- Training Utilities ---
def loss_fn(params, apply_fn, x, y):
    preds = apply_fn({'params': params}, x)
    return jnp.mean((preds - y) ** 2)

def train_step(state, apply_fn, x, y):
    grads = jax.grad(loss_fn)(state.params, apply_fn, x, y)
    return state.apply_gradients(grads=grads)

# --- Training Loop ---
def train_model(model, X_train, Y_train, X_test, Y_test, epochs=50, lr=1e-3):
    rng = jax.random.PRNGKey(0)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(rng, X_train[:1])['params'],
        tx=optax.adam(lr)
    )
    for epoch in range(epochs):
        state = train_step(state, model.apply, X_train, Y_train)
        if epoch % 10 == 0 or epoch == epochs - 1:
            train_loss = loss_fn(state.params, model.apply, X_train, Y_train)
            test_loss = loss_fn(state.params, model.apply, X_test, Y_test)
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    return state

# --- Main Execution ---
def main():
    symbol = input("Enter crypto symbol (e.g., BTC): ").upper()
    horizon = int(input("Enter prediction horizon (days): "))
    df = fetch_data(symbol)
    df = compute_indicators(df)
    # Pass the horizon parameter to prepare_dataset
    X, Y = prepare_dataset(df, horizon=horizon)
    split_idx = int(0.8 * len(X))
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    model = LSTM()
    state = train_model(model, X_train, Y_train, X_test, Y_test)
    preds = model.apply({'params': state.params}, X_test)
    plt.figure(figsize=(12, 6))
    plt.plot(Y_test, label="Actual", alpha=0.7)
    plt.plot(preds, label="Predicted", alpha=0.7)
    plt.title(f"{symbol} Price Prediction ({horizon} day horizon)")
    plt.xlabel("Time Steps")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{symbol}_prediction.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved as {symbol}_prediction.png")

if __name__ == "__main__":
    main()