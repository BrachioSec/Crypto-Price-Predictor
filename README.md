# 💹 Crypto Price Predictor with GUI (JAX + Flax + LSTM)

This project is an advanced cryptocurrency price predictor using deep learning (LSTM), powered by **JAX**, **Flax**, and **Optax**. It includes a user-friendly **Tkinter GUI**, letting users input a crypto symbol and prediction horizon, then displaying actual vs predicted prices in a clean interactive chart.

---

## ✨ Features

* 📈 Real-time historical data fetch via [Yahoo Finance](https://finance.yahoo.com)
* 🔧 Technical indicator extraction (MACD, RSI, EMA, SMA, Bollinger Bands)
* 🧠 JAX-accelerated LSTM neural network built with Flax
* 📊 Interactive GUI (via Tkinter) for symbol input and prediction
* 📉 Live Matplotlib graph in GUI canvas (no terminal required)

---

## 🛠️ Requirements

### 🔢 Python Version

* Python **3.9**, **3.10** or **3.11** (recommended)
* ❌ Python 3.12+ not yet fully supported by JAX/Flax

### 📦 Dependencies (install via pip)

```bash
pip install jax[cpu] flax optax yfinance pandas numpy matplotlib
```

> 💡 On Apple M1/M2 Macs:

```bash
pip install --upgrade "jax[cpu]"
```

Or use `requirements.txt`:

```txt
jax[cpu]
flax
optax
yfinance
pandas
numpy
matplotlib
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your_username/crypto-price-predictor.git
cd crypto-price-predictor
```

2. (Optional) Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
python main.py
```

---

## 🖥️ GUI Preview

* Enter symbol (e.g., `BTC`)
* Set horizon (e.g., `10` days)
* Click "Predict"
* View the prediction chart side-by-side with real prices

---

## 🧠 Model Architecture

* LSTM with JAX and Flax
* Trained on 60-day windows with configurable future horizon
* Uses Optax (Adam optimizer)
* All computation is JIT compiled for speed

---

## 📈 Example Indicators

* RSI (Relative Strength Index)
* MACD (Moving Average Convergence Divergence)
* SMA & EMA
* Bollinger Bands

---

## ⚠️ Disclaimer

This project is **for educational use only**. It is **not intended for financial advice or trading purposes**.

Use responsibly and only on sandbox environments.

---

## 📄 License

MIT License
