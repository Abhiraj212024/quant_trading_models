"""
data_collector.py - Fetch and prepare stock data
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
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
    
    def get_fundamentals(self) -> pd.DataFrame:
        """Fetch fundamental data for clustering"""
        fundamentals = []
        
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
                    'pe_ratio': info.get('trailingPE', np.nan)
                })
            except:
                fundamentals.append({
                    'ticker': ticker,
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': 0,
                    'beta': 1.0,
                    'pe_ratio': np.nan
                })
        
        return pd.DataFrame(fundamentals)
    
    def align_data(self) -> pd.DataFrame:
        """Align all stock data to common timestamps"""
        if not self.data:
            raise ValueError("No data fetched. Run fetch_data() first.")
        
        # Get common dates
        all_dates = set.intersection(*[set(df.index) for df in self.data.values()])
        common_dates = sorted(list(all_dates))
        
        print(f"\nAligned data: {len(common_dates)} common trading days")
        
        # Create aligned dataset with close prices
        aligned = pd.DataFrame(index=common_dates)
        for ticker, df in self.data.items():
            aligned[ticker] = df.loc[common_dates, 'close']
        
        return aligned.sort_index()
    
    def save_data(self, filepath: str = 'data/'):
        """Save collected data"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # Save individual stock data
        for ticker, df in self.data.items():
            df.to_csv(f"{filepath}{ticker}.csv")
        
        # Save aligned data
        aligned = self.align_data()
        aligned.to_csv(f"{filepath}aligned_prices.csv")
        
        # Save fundamentals
        fundamentals = self.get_fundamentals()
        fundamentals.to_csv(f"{filepath}fundamentals.csv", index=False)
        
        print(f"\nData saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
        'JPM', 'BAC', 'GS', 'MS', 'C',             # Finance
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',         # Energy
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'         # Healthcare
    ]
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years
    
    collector = DataCollector(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    collector.fetch_data()
    collector.save_data()