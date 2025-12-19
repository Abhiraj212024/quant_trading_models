"""
data_collector.py - Enhanced stock data collector with ML features
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataCollector:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        
    def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for all tickers"""
        print("Fetching stock data...")
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                if len(df) > 0:
                    df.columns = [col.lower() for col in df.columns]
                    self.data[ticker] = df
                    print(f"✓ {ticker}: {len(df)} records")
                else:
                    print(f"✗ {ticker}: No data available")
            except Exception as e:
                print(f"✗ {ticker}: Error - {str(e)}")
        
        return self.data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Volatility
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'atr_{window}'] = self._calculate_atr(df, window)
        
        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_28'] = self._calculate_rsi(df['close'], 28)
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        for window in [20]:
            sma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma
            df[f'bb_position_{window}'] = (df['close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        return df
    
    def add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate order book features (real data requires market data feed)
        In production, replace with actual order book data
        """
        # Bid-Ask Spread proxy (using high-low as approximation)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        
        # Volume imbalance (approximation using volume and price movement)
        df['volume_imbalance'] = np.where(
            df['close'] > df['open'],
            df['volume'],
            -df['volume']
        )
        df['volume_imbalance_ratio'] = df['volume_imbalance'] / df['volume']
        
        # Rolling volume imbalance
        for window in [5, 10, 20]:
            df[f'cum_volume_imbalance_{window}'] = df['volume_imbalance'].rolling(window).sum()
            df[f'volume_imbalance_ratio_{window}'] = df[f'cum_volume_imbalance_{window}'] / df['volume'].rolling(window).sum()
        
        # Price pressure (rate of price change weighted by volume)
        df['price_pressure'] = df['returns'] * df['volume_ratio']
        df['price_pressure_5'] = df['price_pressure'].rolling(5).mean()
        
        # Liquidity proxy
        df['liquidity_proxy'] = df['volume'] / (df['high'] - df['low'])
        
        return df
    
    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction targets for 5, 10, 15, 20 days ahead"""
        for horizon in [5, 10, 15, 20]:
            # Price direction (binary classification)
            df[f'target_direction_{horizon}d'] = np.where(
                df['close'].shift(-horizon) > df['close'],
                1,  # Up
                0   # Down/Flat
            )
            
            # Returns (regression target)
            df[f'target_returns_{horizon}d'] = (
                df['close'].shift(-horizon) / df['close'] - 1
            )
            
            # Volatility target (for risk estimation)
            df[f'target_volatility_{horizon}d'] = (
                df['returns'].shift(-horizon).rolling(horizon).std()
            )
            
            # Maximum favorable excursion (MFE) and adverse excursion (MAE)
            future_highs = df['high'].rolling(horizon).max().shift(-horizon)
            future_lows = df['low'].rolling(horizon).min().shift(-horizon)
            df[f'target_mfe_{horizon}d'] = (future_highs / df['close'] - 1)
            df[f'target_mae_{horizon}d'] = (future_lows / df['close'] - 1)
        
        return df
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        df['is_quarter_start'] = df.index.is_quarter_start.astype(int)
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        return df
    
    def add_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Add lagged features for sequence models"""
        base_features = ['returns', 'volume_ratio', 'rsi_14', 'macd', 'volatility_10']
        
        for feature in base_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr
    
    def process_all_features(self) -> Dict[str, pd.DataFrame]:
        """Process all features for all tickers"""
        print("\nProcessing features...")
        processed_data = {}
        
        for ticker, df in self.data.items():
            print(f"Processing {ticker}...")
            df_processed = df.copy()
            
            # Add all feature categories
            df_processed = self.add_technical_indicators(df_processed)
            df_processed = self.add_orderbook_features(df_processed)
            df_processed = self.add_temporal_features(df_processed)
            df_processed = self.add_lagged_features(df_processed)
            df_processed = self.add_target_variables(df_processed)
            
            # Remove NaN rows (from indicators and targets)
            df_processed = df_processed.dropna()
            
            processed_data[ticker] = df_processed
            print(f"  {ticker}: {len(df_processed)} samples, {len(df_processed.columns)} features")
        
        return processed_data
    
    def get_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental data for clustering"""
        fundamentals = []
        
        print("\nFetching fundamentals...")
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                fundamentals.append({
                    'ticker': ticker,
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0),
                    'pe_ratio': info.get('trailingPE', np.nan),
                    'forward_pe': info.get('forwardPE', np.nan),
                    'peg_ratio': info.get('pegRatio', np.nan),
                    'price_to_book': info.get('priceToBook', np.nan),
                    'profit_margin': info.get('profitMargins', np.nan),
                    'debt_to_equity': info.get('debtToEquity', np.nan),
                })
                print(f"  ✓ {ticker}")
            except Exception as e:
                print(f"  ✗ {ticker}: {str(e)}")
                fundamentals.append({
                    'ticker': ticker,
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': 0,
                    'beta': 1.0,
                    'pe_ratio': np.nan,
                    'forward_pe': np.nan,
                    'peg_ratio': np.nan,
                    'price_to_book': np.nan,
                    'profit_margin': np.nan,
                    'debt_to_equity': np.nan,
                })
        
        return pd.DataFrame(fundamentals)
    
    def get_clustering_features(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract features for clustering stocks"""
        clustering_features = []
        
        print("\nCalculating clustering features...")
        for ticker, df in processed_data.items():
            features = {
                'ticker': ticker,
                # Return statistics
                'mean_returns': df['returns'].mean(),
                'std_returns': df['returns'].std(),
                'skew_returns': df['returns'].skew(),
                'kurt_returns': df['returns'].kurtosis(),
                # Volatility patterns
                'mean_volatility': df['volatility_20'].mean(),
                'volatility_of_volatility': df['volatility_20'].std(),
                # Momentum
                'mean_rsi': df['rsi_14'].mean(),
                'trend_strength': df['sma_50'].iloc[-1] / df['sma_200'].iloc[-1] - 1 if len(df) > 200 else 0,
                # Volume
                'mean_volume_ratio': df['volume_ratio'].mean(),
                'volume_volatility': df['volume_ratio'].std(),
                # Liquidity
                'mean_spread': df['spread_proxy'].mean(),
            }
            clustering_features.append(features)
        
        return pd.DataFrame(clustering_features)
    
    def save_data(self, processed_data: Dict[str, pd.DataFrame], filepath: str = 'data/'):
        """Save all processed data"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        print(f"\nSaving data to {filepath}...")
        
        # Save individual stock data with features
        for ticker, df in processed_data.items():
            df.to_csv(f"{filepath}{ticker}_features.csv")
        
        # Save fundamentals
        fundamentals = self.get_fundamentals()
        fundamentals.to_csv(f"{filepath}fundamentals.csv", index=False)
        
        # Save clustering features
        clustering_features = self.get_clustering_features(processed_data)
        clustering_features.to_csv(f"{filepath}clustering_features.csv", index=False)
        
        print(f"✓ Data saved successfully")
        print(f"  - {len(processed_data)} stock feature files")
        print(f"  - Fundamentals data")
        print(f"  - Clustering features")


if __name__ == "__main__":
    # Example usage
    tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',
        # Finance
        'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE', 'SBUX',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'HON'
    ]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years
    
    collector = EnhancedDataCollector(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Fetch and process
    collector.fetch_data()
    processed_data = collector.process_all_features()
    collector.save_data(processed_data)
    
    print("\n" + "="*50)
    print("Data collection complete!")
    print("="*50)