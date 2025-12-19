"""
main_pipeline.py - Complete ML trading system pipeline
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collector import EnhancedDataCollector
from stock_clustering import StockClusterer
from model_training import EnsembleModel
from stochastic_methods import EnsembleProbability
from backtester import Backtester, BacktestConfig

import torch
print(f"PyTorch available: {torch.cuda.is_available()}")


class TradingPipeline:
    """End-to-end ML trading pipeline"""
    
    def __init__(self, tickers: list, start_date: str, end_date: str,
                 output_dir: str = '/kaggle/working/output/'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/data", exist_ok=True)
        os.makedirs(f"{output_dir}/clusters", exist_ok=True)
        os.makedirs(f"{output_dir}/models", exist_ok=True)
        os.makedirs(f"{output_dir}/results", exist_ok=True)
        
        # Pipeline components
        self.collector = None
        self.clusterer = None
        self.models = {}
        self.stochastic_models = {}
        
    def step1_collect_data(self):
        """Step 1: Data collection and feature engineering"""
        print("\n" + "="*60)
        print("STEP 1: DATA COLLECTION & FEATURE ENGINEERING")
        print("="*60)
        
        self.collector = EnhancedDataCollector(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        # Fetch raw data
        self.collector.fetch_data()
        
        # Process features
        processed_data = self.collector.process_all_features()
        
        # Save data
        self.collector.save_data(processed_data, f"{self.output_dir}/data/")
        
        return processed_data
    
    def step2_cluster_stocks(self, processed_data: dict, n_clusters: int = 5):
        """Step 2: Cluster stocks by similarity"""
        print("\n" + "="*60)
        print("STEP 2: STOCK CLUSTERING")
        print("="*60)
        
        # Load fundamentals and clustering features
        fundamentals = pd.read_csv(f"{self.output_dir}/data/fundamentals.csv")
        clustering_features = pd.read_csv(f"{self.output_dir}/data/clustering_features.csv")
        
        # Create clusterer
        self.clusterer = StockClusterer(
            processed_data=processed_data,
            fundamentals=fundamentals,
            clustering_features=clustering_features
        )
        
        # Perform hybrid clustering
        cluster_results = self.clusterer.hybrid_clustering(n_clusters=n_clusters)
        
        # Visualize and save
        self.clusterer.visualize_clusters(cluster_results, 
                                         save_path=f"{self.output_dir}/clusters/")
        self.clusterer.save_clusters(f"{self.output_dir}/clusters/")
        
        return cluster_results
    
    def step3_train_models(self, cluster_results: dict, horizons: list = [5, 10, 15]):
        """Step 3: Train ensemble models for each cluster"""
        print("\n" + "="*60)
        print("STEP 3: TRAINING ML MODELS")
        print("="*60)
        
        for horizon in horizons:
            print(f"\n--- Training models for {horizon}-day horizon ---")
            
            for cluster_id in np.unique(cluster_results['labels']):
                print(f"\nCluster {cluster_id}:")
                
                # Get cluster data
                cluster_data = self.clusterer.get_cluster_data(cluster_id)
                
                if len(cluster_data) < 3:
                    print(f"  Skipping (too few stocks)")
                    continue
                
                # Create ensemble model
                model = EnsembleModel(cluster_id=cluster_id, horizon=horizon)
                
                try:
                    # Prepare data
                    X_train, X_val, y_train, y_val = model.prepare_data(cluster_data)
                    
                    # Train models
                    model.train_lightgbm(X_train, X_val, y_train, y_val)
                    model.train_cnn_lstm(X_train, X_val, y_train, y_val)
                    model.train_transformer(X_train, X_val, y_train, y_val)
                    
                    # Save model
                    model_path = f"{self.output_dir}/models/cluster_{cluster_id}_horizon_{horizon}d"
                    model.save(model_path)
                    
                    # Store in dictionary
                    self.models[(cluster_id, horizon)] = model
                    
                    print(f"✓ Models trained and saved")
                    
                except Exception as e:
                    print(f"✗ Error training models: {str(e)}")
                    continue
    
    def step4_generate_signals(self, cluster_results: dict, 
                               processed_data: dict, horizon: int = 5) -> dict:
        """Step 4: Generate trading signals with probabilities"""
        print("\n" + "="*60)
        print("STEP 4: GENERATING TRADING SIGNALS")
        print("="*60)
        
        all_signals = {}
        
        for cluster_id in np.unique(cluster_results['labels']):
            # Get cluster stocks
            cluster_tickers = [
                ticker for ticker, label in zip(cluster_results['tickers'], 
                                               cluster_results['labels'])
                if label == cluster_id
            ]
            
            # Load model
            if (cluster_id, horizon) not in self.models:
                print(f"Cluster {cluster_id}: No model found")
                continue
            
            model = self.models[(cluster_id, horizon)]
            
            for ticker in cluster_tickers:
                if ticker not in processed_data:
                    continue
                
                try:
                    df = processed_data[ticker]
                    
                    # Get recent data for prediction
                    recent_data = df.tail(100)[model.feature_names].values
                    
                    # ML prediction
                    ml_predictions = model.predict(recent_data)
                    
                    # Stochastic probability estimation
                    returns = df['returns'].tail(252)
                    stoch_model = EnsembleProbability(returns)
                    
                    # Generate signals for each day
                    signals_list = []
                    for i in range(len(ml_predictions)):
                        if i >= len(df) - len(ml_predictions):
                            date_idx = len(df) - len(ml_predictions) + i
                            current_price = df.iloc[date_idx]['close']
                            ml_pred = ml_predictions[i]
                            
                            signal_info = stoch_model.generate_trading_signal(
                                current_price=current_price,
                                horizon=horizon,
                                ml_prediction=ml_pred,
                                threshold=0.6
                            )
                            
                            signals_list.append({
                                'date': df.index[date_idx],
                                'signal': 1 if signal_info['action'] == 'BUY' else -1 if signal_info['action'] == 'SELL' else 0,
                                'confidence': signal_info['confidence'],
                                'probability_up': signal_info['probability_up'],
                                'expected_return': signal_info['expected_return'],
                                'var_95': signal_info['risk_var_95'],
                                'kelly_size': signal_info['position_size_kelly']
                            })
                    
                    if signals_list:
                        all_signals[ticker] = pd.DataFrame(signals_list).set_index('date')
                        print(f"  {ticker}: {len(signals_list)} signals generated")
                
                except Exception as e:
                    print(f"  {ticker}: Error generating signals - {str(e)}")
                    continue
        
        # Save signals
        signals_dir = f"{self.output_dir}/results/signals/"
        os.makedirs(signals_dir, exist_ok=True)
        for ticker, signals_df in all_signals.items():
            signals_df.to_csv(f"{signals_dir}{ticker}_signals.csv")
        
        print(f"\n✓ Signals generated for {len(all_signals)} stocks")
        return all_signals
    
    def step5_backtest(self, processed_data: dict, signals: dict):
        """Step 5: Backtest trading strategy"""
        print("\n" + "="*60)
        print("STEP 5: BACKTESTING STRATEGY")
        print("="*60)
        
        # Configure backtester
        config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            max_positions=10,
            stop_loss=0.05,
            take_profit=0.15
        )
        
        backtester = Backtester(config)
        
        # Prepare data (use only 'close' for backtester)
        price_data = {
            ticker: df[['open', 'high', 'low', 'close', 'volume']]
            for ticker, df in processed_data.items()
        }
        
        # Run backtest
        results = backtester.run(
            data=price_data,
            signals=signals,
            start_date=None,  # Use all available data
            end_date=None
        )
        
        # Display results
        backtester.print_results()
        
        # Plot results
        backtester.plot_results(save_path=f"{self.output_dir}/results/")
        
        # Export trades
        backtester.export_trades(f"{self.output_dir}/results/trades.csv")
        
        return results
    
    def run_full_pipeline(self, n_clusters: int = 5, horizons: list = [5, 10]):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print(" "*15 + "ML TRADING SYSTEM PIPELINE")
        print("="*70)
        print(f"Tickers: {len(self.tickers)}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Output directory: {self.output_dir}")
        print("="*70)
        
        # Step 1: Data Collection
        processed_data = self.step1_collect_data()
        
        # Step 2: Clustering
        cluster_results = self.step2_cluster_stocks(processed_data, n_clusters)
        
        # Step 3: Train Models
        self.step3_train_models(cluster_results, horizons)
        
        # Step 4: Generate Signals (use first horizon for backtesting)
        signals = self.step4_generate_signals(cluster_results, processed_data, 
                                              horizon=horizons[0])
        
        # Step 5: Backtest
        backtest_results = self.step5_backtest(processed_data, signals)
        
        print("\n" + "="*70)
        print(" "*20 + "PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nResults saved to: {self.output_dir}")
        
        return {
            'processed_data': processed_data,
            'clusters': cluster_results,
            'models': self.models,
            'signals': signals,
            'backtest': backtest_results
        }


def main():
    """Example usage"""
    
    # Define stock universe
    tickers = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        # Finance
        'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'TMO',
        # Consumer
        'WMT', 'HD', 'MCD', 'NKE',
        # Industrial
        'BA', 'CAT', 'GE'
    ]
    
    # Define date range (5 years of data)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)
    
    # Create pipeline
    pipeline = TradingPipeline(
        tickers=tickers,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        output_dir='ml_trading_output/'
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(
        n_clusters=4,
        horizons=[5, 10]  # 5-day and 10-day predictions
    )
    
    print("\n" + "="*70)
    print("Pipeline execution summary:")
    print(f"  - Data collected for {len(results['processed_data'])} stocks")
    print(f"  - Formed {len(np.unique(results['clusters']['labels']))} clusters")
    print(f"  - Trained {len(results['models'])} ensemble models")
    print(f"  - Generated signals for {len(results['signals'])} stocks")
    print(f"  - Backtest return: {results['backtest']['total_return_pct']:.2f}%")
    print(f"  - Sharpe ratio: {results['backtest']['sharpe_ratio']:.2f}")
    print("="*70)


if __name__ == "__main__":
    main()