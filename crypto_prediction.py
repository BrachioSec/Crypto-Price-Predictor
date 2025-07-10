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
    """Fetch recent cryptocurrency data with robust error handling"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # List of possible ticker formats to try
    possible_tickers = [
        f"{symbol.upper()}-USD",
        f"{symbol.upper()}USD",
        f"{symbol.upper()}-USDT",
        symbol.upper()
    ]
    
    df = None
    successful_ticker = None
    
    print(f"Attempting to fetch {symbol} data from {start_date} to {end_date}...")
    
    for ticker in possible_tickers:
        try:
            print(f"Trying ticker: {ticker}")
            temp_df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if temp_df is not None and len(temp_df) > 100:  # Ensure we have sufficient data
                df = temp_df
                successful_ticker = ticker
                print(f"✅ Successfully downloaded {len(df)} days of data using {ticker}")
                break
            else:
                print(f"❌ No data or insufficient data for {ticker}")
                
        except Exception as e:
            print(f"❌ Error with {ticker}: {str(e)}")
            continue
    
    if df is None or len(df) < 100:
        # Try some popular symbols as examples
        popular_symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'MATIC-USD']
        print(f"\n❌ Could not fetch data for {symbol}")
        print(f"💡 Try one of these popular symbols: {', '.join([s.split('-')[0] for s in popular_symbols])}")
        print(f"💡 Or check if the symbol is correct on Yahoo Finance")
        raise ValueError(f"No data available for symbol: {symbol}")
    
    # Handle MultiIndex columns by flattening them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure all required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    df = df[required_columns].dropna()
    
    if len(df) < 100:
        raise ValueError(f"Insufficient data after cleaning: {len(df)} days (need at least 100)")
    
    print(f"✅ Final dataset: {len(df)} days of clean data")
    return df, successful_ticker

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
    """Compute technical indicators with robust error handling"""
    print("Computing technical indicators...")
    
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    
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
    df['Volume_SMA'] = volume.rolling(window=20).mean()
    df['Volume_ratio'] = volume / df['Volume_SMA']
    
    # Price change indicators
    df['Price_Change'] = close.pct_change()
    df['Volatility'] = close.rolling(window=20).std() / close.rolling(window=20).mean()
    
    # High-Low spread
    df['HL_spread'] = (high - low) / close
    
    # Clean the dataframe
    df_clean = df.dropna()
    
    if len(df_clean) < 50:
        raise ValueError(f"Insufficient data after computing indicators: {len(df_clean)} days")
    
    print(f"✅ Computed indicators for {len(df_clean)} days")
    return df_clean

# --- Dataset Preparation with Better Validation ---
def prepare_dataset(df, target_col='Close', window=30, horizon=1):
    """Prepare dataset with proper validation and normalization"""
    print(f"Preparing dataset with window={window}, horizon={horizon}")
    
    if len(df) < window + horizon + 10:
        raise ValueError(f"Insufficient data: {len(df)} days. Need at least {window + horizon + 10} days.")
    
    # Create a copy and reset index
    df_copy = df.copy().reset_index(drop=True)
    
    # Select features (exclude target and date columns)
    feature_cols = [col for col in df_copy.select_dtypes(include=[np.number]).columns 
                   if col != target_col]
    
    print(f"Selected features: {feature_cols}")
    
    # Separate features and target
    features = df_copy[feature_cols]
    target = df_copy[target_col].values
    
    # Validate features
    if features.empty or len(features) == 0:
        raise ValueError("No features available for training")
    
    # Check for infinite or NaN values
    if features.isnull().any().any() or np.isinf(features.values).any():
        print("⚠️ Found NaN or infinite values, cleaning...")
        features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    
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
    
    if len(X) == 0:
        raise ValueError("No sequences could be created. Check window and horizon parameters.")
    
    X = jnp.array(X, dtype=jnp.float32)
    Y = jnp.array(Y, dtype=jnp.float32)
    
    print(f"✅ Created {len(X)} sequences with {X.shape[2]} features")
    
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
    
    print("🚀 Starting training...")
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
                print(f"⏰ Early stopping at epoch {epoch+1}")
                break
    
    return state

# --- Enhanced Evaluation and Visualization ---
def evaluate_model(state, model, X_test, Y_test, target_scaler, symbol, horizon, ticker_used):
    """Evaluate model and create beautiful, readable predictions"""
    # Get predictions (normalized)
    preds_norm = model.apply({'params': state.params}, X_test, training=False)
    
    # Denormalize predictions and targets
    preds_real = target_scaler.inverse_transform(preds_norm.reshape(-1, 1)).flatten()
    y_test_real = target_scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = np.mean((preds_real - y_test_real) ** 2)
    mae = np.mean(np.abs(preds_real - y_test_real))
    mape = np.mean(np.abs((y_test_real - preds_real) / y_test_real)) * 100
    
    print(f"\n📊 Model Evaluation for {symbol}:")
    print(f"   MSE: ${mse:,.2f}")
    print(f"   MAE: ${mae:,.2f}")
    print(f"   MAPE: {mape:.2f}%")
    
    # Create beautiful plots
    fig = plt.figure(figsize=(20, 12))
    
    # Main prediction plot
    ax1 = plt.subplot(3, 2, (1, 4))
    plot_points = min(100, len(y_test_real))
    x_axis = range(plot_points)
    
    # Plot with better styling
    plt.plot(x_axis, y_test_real[-plot_points:], 'b-', label="Actual Price", linewidth=3, alpha=0.8)
    plt.plot(x_axis, preds_real[-plot_points:], 'r--', label="Predicted Price", linewidth=3, alpha=0.8)
    
    plt.title(f'🚀 {symbol} Price Prediction ({horizon} day horizon)\nUsing ticker: {ticker_used}', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time Steps (Most Recent Predictions)', fontsize=12, fontweight='bold')
    plt.ylabel('Price (USD)', fontsize=12, fontweight='bold')
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add price annotations
    plt.annotate(f'Latest Actual: ${y_test_real[-1]:,.2f}', 
                xy=(plot_points-1, y_test_real[-1]), xytext=(plot_points-20, y_test_real[-1]*1.05),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                fontsize=10, color='blue', fontweight='bold')
    
    plt.annotate(f'Latest Predicted: ${preds_real[-1]:,.2f}', 
                xy=(plot_points-1, preds_real[-1]), xytext=(plot_points-20, preds_real[-1]*0.95),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', fontweight='bold')
    
    # Error analysis plot
    ax2 = plt.subplot(3, 2, 5)
    error = y_test_real[-plot_points:] - preds_real[-plot_points:]
    colors = ['red' if e > 0 else 'green' for e in error]
    plt.bar(x_axis, error, color=colors, alpha=0.6, width=0.8)
    plt.title('📈 Prediction Error Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Error (USD)', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Metrics summary plot
    ax3 = plt.subplot(3, 2, 6)
    metrics = ['MAPE (%)', 'MAE ($)', 'Error %']
    values = [mape, mae/1000, abs(y_test_real[-1] - preds_real[-1])/y_test_real[-1]*100]
    colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('📊 Performance Metrics', fontsize=14, fontweight='bold')
    plt.ylabel('Value', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Overall layout adjustments
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    filename = f'{symbol}_enhanced_prediction.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"📁 Enhanced prediction plot saved as {filename}")
    plt.close()
    
    return preds_real, y_test_real

# --- Main Execution with Enhanced UX ---
def main():
    print("🚀 Cryptocurrency Price Prediction Model")
    print("=" * 50)
    
    while True:
        symbol = input("\n💰 Enter crypto symbol (e.g., BTC, ETH, ADA): ").strip().upper()
        if symbol:
            break
        print("❌ Please enter a valid symbol")
    
    while True:
        try:
            horizon = int(input("📅 Enter prediction horizon (days, 1-7 recommended): "))
            if 1 <= horizon <= 30:
                break
            elif horizon > 30:
                print("⚠️ Warning: Large horizons (>30 days) may not be realistic for crypto prediction")
                confirm = input("Continue anyway? (y/n): ").lower()
                if confirm == 'y':
                    break
            else:
                print("❌ Please enter a positive number")
        except ValueError:
            print("❌ Please enter a valid number")
    
    try:
        # Fetch and prepare data
        df, ticker_used = fetch_data(symbol, days=730)
        df = compute_indicators(df)
        
        # Prepare dataset with normalization
        X, Y, feature_scaler, target_scaler = prepare_dataset(df, horizon=horizon, window=30)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, Y_train = X[:split_idx], Y[:split_idx]
        X_test, Y_test = X[split_idx:], Y[split_idx:]
        
        print(f"\n📊 Dataset Information:")
        print(f"   Training set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        print(f"   Features: {X.shape[2]} technical indicators")
        
        # Create and train model
        model = ImprovedLSTM()
        state = train_model(model, X_train, Y_train, X_test, Y_test, epochs=100, lr=0.001)
        
        # Evaluate model
        preds_real, y_test_real = evaluate_model(state, model, X_test, Y_test, target_scaler, symbol, horizon, ticker_used)
        
        print(f"\n🎯 Final Results:")
        print(f"   Latest actual price: ${y_test_real[-1]:,.2f}")
        print(f"   Latest predicted price: ${preds_real[-1]:,.2f}")
        price_diff = abs(y_test_real[-1] - preds_real[-1])
        accuracy = (1 - price_diff/y_test_real[-1]) * 100
        print(f"   Price difference: ${price_diff:,.2f}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\n💡 Suggestions:")
        print("   • Check if the cryptocurrency symbol is correct")
        print("   • Try popular symbols: BTC, ETH, ADA, DOT, MATIC")
        print("   • Ensure you have internet connection")
        print("   • Wait a moment and try again (API rate limits)")

if __name__ == "__main__":
    main()
