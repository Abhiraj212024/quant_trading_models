# 📈 Quantitative Trading System  
**End-to-End Machine Learning Pipeline for Financial Time-Series Research**

This project is a full-stack quantitative research and trading system that covers the entire lifecycle of ML-driven trading strategies — from **market data collection and regime discovery** to **deep learning modeling, probabilistic signal generation, and portfolio backtesting**.

It is designed as a **research framework**, focusing on realism:  
✔ multi-asset data  
✔ regime clustering  
✔ deep sequence models  
✔ transaction costs  
✔ risk-adjusted evaluation  

---

## 🚀 Key Features

### 🔗 End-to-End Quant Research Pipeline
- Automated workflow covering:
  - Data collection  
  - Feature engineering  
  - Stock/regime clustering  
  - Dataset generation  
  - Model training  
  - Signal generation  
  - Portfolio backtesting  
  - Performance & risk analysis  

---

### 🗄️ Market Data Engineering
- Multi-asset OHLCV data collection
- Rolling-window supervised dataset construction
- Feature scaling, cleaning, and alignment
- Leakage-aware train/validation splits

---

### 🧩 Regime & Stock Clustering
- Statistical stock-level feature aggregation
- Dimensionality reduction and unsupervised learning
- K-Means clustering to group stocks into regimes
- Cluster-specialized modeling

---

### 🧠 Deep Learning for Time-Series
Implemented and benchmarked multiple architectures:

- Temporal Convolutional Networks (TCN)  
- CNN–LSTM hybrid models  
- Transformer-based sequence models  

Supports:
- Multi-horizon forecasting  
- Probabilistic prediction outputs  
- Model comparison and benchmarking

---

### 📊 Probabilistic Signal Modeling
- Probability-based trade signals
- Confidence thresholding
- Prediction distribution analysis
- Calibration error measurement

---

### ⚙️ Portfolio Backtesting Engine
- Full trade lifecycle simulation
- Position management
- Commission and cost modeling
- Automated computation of:
  - CAGR  
  - Sharpe & Sortino ratios  
  - Maximum drawdown  
  - Win rate & profit factor  
- Equity curves, diagnostics, and trade logs

---

## 📈 Example Results

From one experimental run:

- **Total return:** 24.31%  
- **CAGR:** 7.56%  
- **Sharpe ratio:** 0.83  
- **Max drawdown:** –7.54%  
- **Trades executed:** 165  
- **Assets used:** 147 stocks  
- **Models trained:** 10  

The system also generates probability calibration plots, performance diagnostics, and exports all trades for further analysis.

---

## 🏗️ System Architecture
Data Collection
↓
Feature Engineering & Cleaning
↓
Stock / Regime Clustering
↓
Dataset Construction
↓
Deep Learning Model Training
↓
Probabilistic Signal Generation
↓
Portfolio Backtesting Engine
↓
Risk Metrics, Plots & Trade Logs


---

## 🛠️ Tech Stack

- **Languages:** Python
- **ML / DL:** TensorFlow / Keras, NumPy, Pandas, Scikit-learn
- **Time-Series Models:** TCN, CNN-LSTM, Transformer
- **Visualization:** Matplotlib, Plotly
- **Finance & Research:** Custom backtester, probabilistic modeling, clustering

---


