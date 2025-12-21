"""
data_collector.py - Enhanced stock data collector with ML features
FIXED: No look-ahead bias, expanded stock universe
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
        successful = 0
        failed = 0
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                if len(df) > 100:  # Minimum data requirement
                    df.columns = [col.lower() for col in df.columns]
                    self.data[ticker] = df
                    successful += 1
                    if successful % 10 == 0:
                        print(f"  Fetched {successful} stocks...")
                else:
                    failed += 1
            except Exception as e:
                failed += 1
        
        print(f"✓ Successfully fetched {successful} stocks")
        print(f"✗ Failed to fetch {failed} stocks")
        return self.data
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators - NO LOOK-AHEAD BIAS"""
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving averages (use only past data)
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Volatility (use only past data)
        for window in [5, 10, 20, 60]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'atr_{window}'] = self._calculate_atr(df, window)
        
        # Momentum indicators
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_28'] = self._calculate_rsi(df['close'], 28)
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
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
        df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, 1)
        df['obv'] = (np.sign(df['returns']) * df['volume']).fillna(0).cumsum()
        
        # Price patterns
        df['high_low_ratio'] = df['high'] / df['low'].replace(0, 1)
        df['close_open_ratio'] = df['close'] / df['open'].replace(0, 1)
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        return df
    
    def add_orderbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate order book features - NO LOOK-AHEAD BIAS"""
        # Bid-Ask Spread proxy (using high-low as approximation)
        df['spread_proxy'] = (df['high'] - df['low']) / df['close'].replace(0, 1)
        
        # Volume imbalance
        df['volume_imbalance'] = np.where(
            df['close'] > df['open'],
            df['volume'],
            -df['volume']
        )
        df['volume_imbalance_ratio'] = df['volume_imbalance'] / df['volume'].replace(0, 1)
        
        # Rolling volume imbalance
        for window in [5, 10, 20]:
            df[f'cum_volume_imbalance_{window}'] = df['volume_imbalance'].rolling(window).sum()
            vol_sum = df['volume'].rolling(window).sum().replace(0, 1)
            df[f'volume_imbalance_ratio_{window}'] = df[f'cum_volume_imbalance_{window}'] / vol_sum
        
        # Price pressure
        df['price_pressure'] = df['returns'] * df['volume_ratio']
        df['price_pressure_5'] = df['price_pressure'].rolling(5).mean()
        
        # Liquidity proxy
        df['liquidity_proxy'] = df['volume'] / (df['high'] - df['low']).replace(0, 1)
        
        return df
    
    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add prediction targets - FUTURE DATA, separated properly"""
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
            
            # Volatility target
            df[f'target_volatility_{horizon}d'] = (
                df['returns'].shift(-horizon).rolling(horizon).std()
            )
            
            # Maximum favorable/adverse excursion
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
        """Add lagged features - NO LOOK-AHEAD BIAS"""
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
        rs = gain / loss.replace(0, 1)
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
            df_processed = df.copy()
            
            # Add all feature categories
            df_processed = self.add_technical_indicators(df_processed)
            df_processed = self.add_orderbook_features(df_processed)
            df_processed = self.add_temporal_features(df_processed)
            df_processed = self.add_lagged_features(df_processed)
            df_processed = self.add_target_variables(df_processed)
            
            # Don't drop NaN yet - we'll handle this properly during train/test split
            processed_data[ticker] = df_processed
        
        print(f"✓ Processed {len(processed_data)} stocks")
        return processed_data
    
    def get_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental data for clustering"""
        fundamentals = []
        
        print("\nFetching fundamentals...")
        for i, ticker in enumerate(self.tickers):
            if i % 10 == 0 and i > 0:
                print(f"  Fetched fundamentals for {i} stocks...")
            
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
            except Exception as e:
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
        
        print(f"✓ Fetched fundamentals for {len(fundamentals)} stocks")
        return pd.DataFrame(fundamentals)
    
    def get_clustering_features(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract features for clustering stocks"""
        clustering_features = []
        
        print("\nCalculating clustering features...")
        for ticker, df in processed_data.items():
            # Only use past data for clustering
            df_clean = df.dropna(subset=['returns', 'volatility_20'])
            
            if len(df_clean) < 100:
                continue
            
            features = {
                'ticker': ticker,
                # Return statistics
                'mean_returns': df_clean['returns'].mean(),
                'std_returns': df_clean['returns'].std(),
                'skew_returns': df_clean['returns'].skew(),
                'kurt_returns': df_clean['returns'].kurtosis(),
                # Volatility patterns
                'mean_volatility': df_clean['volatility_20'].mean(),
                'volatility_of_volatility': df_clean['volatility_20'].std(),
                # Momentum
                'mean_rsi': df_clean['rsi_14'].mean() if 'rsi_14' in df_clean.columns else 50,
                'trend_strength': (df_clean['sma_50'].iloc[-1] / df_clean['sma_200'].iloc[-1] - 1) if len(df_clean) > 200 and 'sma_200' in df_clean.columns else 0,
                # Volume
                'mean_volume_ratio': df_clean['volume_ratio'].mean() if 'volume_ratio' in df_clean.columns else 1,
                'volume_volatility': df_clean['volume_ratio'].std() if 'volume_ratio' in df_clean.columns else 0,
                # Liquidity
                'mean_spread': df_clean['spread_proxy'].mean() if 'spread_proxy' in df_clean.columns else 0,
            }
            clustering_features.append(features)
        
        return pd.DataFrame(clustering_features)
    
    def save_data(self, processed_data: Dict[str, pd.DataFrame], filepath: str = 'data/'):
        """Save all processed data"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        print(f"\nSaving data to {filepath}...")
        
        # Save individual stock data
        for ticker, df in processed_data.items():
            df.to_csv(f"{filepath}{ticker}_features.csv")
        
        # Save fundamentals
        fundamentals = self.get_fundamentals()
        fundamentals.to_csv(f"{filepath}fundamentals.csv", index=False)
        
        # Save clustering features
        clustering_features = self.get_clustering_features(processed_data)
        clustering_features.to_csv(f"{filepath}clustering_features.csv", index=False)
        
        print(f"✓ Data saved successfully")


def get_expanded_stock_universe() -> List[str]:
    """Get expanded list of liquid stocks across sectors"""
    return [
        # Technology (25 stocks)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'ADBE', 'CRM',
        'ORCL', 'CSCO', 'AVGO', 'INTC', 'AMD', 'QCOM', 'TXN', 'AMAT', 'MU',
        'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MRVL', 'NXPI', 'ADI',
        
        # Finance (20 stocks)
        'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'USB',
        'PNC', 'TFC', 'BK', 'STT', 'COF', 'DFS', 'SPGI', 'MCO', 'AON', 'MMC',
        
        # Healthcare (20 stocks)
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR', 'LLY', 'AMGN',
        'GILD', 'CVS', 'CI', 'HUM', 'ISRG', 'SYK', 'BSX', 'MDT', 'ZTS', 'REGN',
        
        # Consumer Discretionary (15 stocks)
        'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'BKNG', 'CMG', 'ORLY',
        'MAR', 'YUM', 'DG', 'ROST', 'DHI', 'LEN',
        
        # Consumer Staples (10 stocks)
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'MDLZ', 'CL', 'KMB', 'GIS', 'HSY',
        
        # Energy (12 stocks)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HES',
        'HAL', 'BKR',
        
        # Industrials (15 stocks)
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'LMT', 'RTX', 'UNP', 'DE', 'MMM',
        'NOC', 'ETN', 'EMR', 'ITW', 'PH',
        
        # Materials (8 stocks)
        'LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'NEM', 'FCX',
        
        # Utilities (8 stocks)
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
        
        # Real Estate (7 stocks)
        'PLD', 'AMT', 'CCI', 'EQIX', 'PSA', 'SPG', 'O',
        
        # Communication Services (10 stocks)
        'DIS', 'CMCSA', 'NFLX', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'ATVI',
    ]


if __name__ == "__main__":
    tickers = get_expanded_stock_universe()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years
    
    collector = EnhancedDataCollector(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    collector.fetch_data()
    processed_data = collector.process_all_features()
    collector.save_data(processed_data)