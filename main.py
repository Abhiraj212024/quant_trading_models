"""
main.py - FIXED signal generation to properly align predictions with dates
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from data_collector import EnhancedDataCollector, get_expanded_stock_universe
from stock_clustering import StockClusterer
from model_training import EnsembleModel
from stochastic_methods import EnsembleProbability
from backtester import Backtester, BacktestConfig


class TradingPipeline:
    """End-to-end ML trading pipeline WITHOUT look-ahead bias"""
    
    def __init__(self, tickers: list, start_date: str, end_date: str,
                 output_dir: str = 'ml_trading_output/'):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = output_dir
        
        # Create output directories
        for subdir in ['data', 'clusters', 'models', 'results', 'figures']:
            os.makedirs(f"{output_dir}/{subdir}", exist_ok=True)
        
        self.collector = None
        self.clusterer = None
        self.models = {}
        self.stochastic_models = {}
        
    def step1_collect_data(self):
        """Step 1: Data collection and feature engineering"""
        print("\n" + "="*70)
        print("STEP 1: DATA COLLECTION & FEATURE ENGINEERING")
        print("="*70)
        
        self.collector = EnhancedDataCollector(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        self.collector.fetch_data()
        processed_data = self.collector.process_all_features()
        self.collector.save_data(processed_data, f"{self.output_dir}/data/")
        
        print(f"\n✓ Collected data for {len(processed_data)} stocks")
        return processed_data
    
    def step2_cluster_stocks(self, processed_data: dict, n_clusters: int = 5):
        """Step 2: Cluster stocks by similarity"""
        print("\n" + "="*70)
        print("STEP 2: STOCK CLUSTERING")
        print("="*70)
        
        fundamentals = pd.read_csv(f"{self.output_dir}/data/fundamentals.csv")
        clustering_features = pd.read_csv(f"{self.output_dir}/data/clustering_features.csv")
        
        self.clusterer = StockClusterer(
            processed_data=processed_data,
            fundamentals=fundamentals,
            clustering_features=clustering_features
        )
        
        cluster_results = self.clusterer.hybrid_clustering(n_clusters=n_clusters)
        self.clusterer.visualize_clusters(cluster_results, 
                                         save_path=f"{self.output_dir}/clusters/")
        self.clusterer.save_clusters(f"{self.output_dir}/clusters/")
        
        return cluster_results
    
    def step3_train_models(self, cluster_results: dict, horizons: list = [5, 10]):
        """Step 3: Train ensemble models for each cluster"""
        print("\n" + "="*70)
        print("STEP 3: TRAINING ML MODELS (NO LOOK-AHEAD BIAS)")
        print("="*70)
        
        trained_count = 0
        failed_count = 0
        
        for horizon in horizons:
            print(f"\n--- Training models for {horizon}-day horizon ---")
            
            for cluster_id in np.unique(cluster_results['labels']):
                print(f"\nCluster {cluster_id}:")
                
                cluster_data = self.clusterer.get_cluster_data(cluster_id)
                
                if len(cluster_data) < 3:
                    print(f"  Skipping (only {len(cluster_data)} stocks)")
                    failed_count += 1
                    continue
                
                model = EnsembleModel(cluster_id=cluster_id, horizon=horizon)
                
                try:
                    X_train, X_val, y_train, y_val = model.prepare_data(cluster_data)
                    
                    model.train_lightgbm(X_train, X_val, y_train, y_val)
                    model.train_cnn_lstm(X_train, X_val, y_train, y_val)
                    model.train_transformer(X_train, X_val, y_train, y_val)
                    
                    model_path = f"{self.output_dir}/models/cluster_{cluster_id}_horizon_{horizon}d"
                    model.save(model_path)
                    
                    self.models[(cluster_id, horizon)] = model
                    trained_count += 1
                    
                    print(f"  ✓ Cluster {cluster_id} models trained successfully")
                    
                except Exception as e:
                    print(f"  ✗ Error training cluster {cluster_id}: {str(e)}")
                    failed_count += 1
                    continue
        
        print(f"\n✓ Successfully trained {trained_count} model ensembles")
        print(f"✗ Failed to train {failed_count} model ensembles")
    
    def step4_generate_signals(self, cluster_results: dict, 
                               processed_data: dict, horizon: int = 5) -> dict:
        """
        Step 4: Generate trading signals with probabilities
        FIXED: Only use data where predictions are valid (after sequence_length)
        """
        print("\n" + "="*70)
        print("STEP 4: GENERATING TRADING SIGNALS")
        print("="*70)
        
        all_signals = {}
        success_count = 0
        fail_count = 0
        
        for cluster_id in np.unique(cluster_results['labels']):
            cluster_tickers = [
                ticker for ticker, label in zip(cluster_results['tickers'], 
                                               cluster_results['labels'])
                if label == cluster_id
            ]
            
            if (cluster_id, horizon) not in self.models:
                print(f"Cluster {cluster_id}: No model found")
                continue
            
            model = self.models[(cluster_id, horizon)]
            
            print(f"\nCluster {cluster_id}: Generating signals for {len(cluster_tickers)} stocks")
            
            for ticker in cluster_tickers:
                if ticker not in processed_data:
                    continue
                
                try:
                    df = processed_data[ticker]
                    
                    # Get features (without target columns)
                    feature_df = df[model.feature_names].dropna()
                    
                    if len(feature_df) < model.sequence_length + 100:
                        fail_count += 1
                        continue
                    
                    # Get all available feature data
                    X_all = feature_df.values
                    all_dates = feature_df.index
                    
                    # CRITICAL FIX: Drop the first sequence_length points
                    # These are ONLY used for building the window, not for predictions
                    # If we have 1056 points and sequence_length=60:
                    # - Points 0-59: Used to build first window (no prediction)
                    # - Points 60-1055: Each gets a prediction (996 predictions)
                    
                    if len(X_all) <= model.sequence_length:
                        fail_count += 1
                        continue
                    
                    # Make predictions on ALL data
                    ml_predictions = model.predict(X_all)
                    
                    # Drop the first sequence_length dates (they don't have predictions)
                    valid_dates = all_dates[model.sequence_length:]
                    
                    # Now match lengths exactly
                    if len(valid_dates) != len(ml_predictions):
                        # Adjust to minimum length
                        min_len = min(len(valid_dates), len(ml_predictions))
                        valid_dates = valid_dates[:min_len]
                        ml_predictions = ml_predictions[:min_len]
                    
                    prediction_dates = valid_dates
                    
                    if len(prediction_dates) == 0:
                        fail_count += 1
                        continue
                    
                    # Stochastic probability estimation
                    returns = df['returns'].dropna().tail(252)
                    if len(returns) < 50:
                        fail_count += 1
                        continue
                    
                    stoch_model = EnsembleProbability(returns)
                    
                    # Generate signals for each prediction
                    signals_list = []
                    
                    for i, date_idx in enumerate(prediction_dates):
                        try:
                            current_price = df.loc[date_idx, 'close']
                            ml_pred = ml_predictions[i]
                            
                            # Clip prediction to valid probability range
                            ml_pred = np.clip(ml_pred, 0.0, 1.0)
                            
                            signal_info = stoch_model.generate_trading_signal(
                                current_price=current_price,
                                horizon=horizon,
                                ml_prediction=ml_pred,
                                threshold=0.6
                            )
                            
                            signals_list.append({
                                'date': date_idx,
                                'signal': 1 if signal_info['action'] == 'BUY' else -1 if signal_info['action'] == 'SELL' else 0,
                                'confidence': signal_info['confidence'],
                                'probability_up': signal_info['probability_up'],
                                'expected_return': signal_info['expected_return'],
                                'var_95': signal_info['risk_var_95'],
                                'kelly_size': signal_info['position_size_kelly']
                            })
                        except (KeyError, IndexError) as e:
                            # Date not found in original dataframe or index error
                            continue
                    
                    if len(signals_list) > 0:
                        all_signals[ticker] = pd.DataFrame(signals_list).set_index('date')
                        success_count += 1
                        if success_count % 20 == 0:
                            print(f"  Generated signals for {success_count} stocks...")
                
                except Exception as e:
                    # Print more detailed error for debugging
                    import traceback
                    print(f"  {ticker}: Error - {str(e)}")
                    # Uncomment for full traceback:
                    # traceback.print_exc()
                    fail_count += 1
                    continue
        
        # Save signals
        signals_dir = f"{self.output_dir}/results/signals/"
        os.makedirs(signals_dir, exist_ok=True)
        for ticker, signals_df in all_signals.items():
            signals_df.to_csv(f"{signals_dir}{ticker}_signals.csv")
        
        print(f"\n✓ Generated signals for {success_count} stocks")
        print(f"✗ Failed to generate signals for {fail_count} stocks")
        return all_signals
    
    def step5_backtest(self, processed_data: dict, signals: dict):
        """Step 5: Backtest trading strategy"""
        print("\n" + "="*70)
        print("STEP 5: BACKTESTING STRATEGY")
        print("="*70)
        
        if len(signals) == 0:
            print("✗ No signals generated, cannot run backtest")
            return {'error': 'No signals'}
        
        config = BacktestConfig(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0005,
            max_positions=10,
            stop_loss=0.05,
            take_profit=0.15
        )
        
        backtester = Backtester(config)
        
        price_data = {
            ticker: df[['open', 'high', 'low', 'close', 'volume']]
            for ticker, df in processed_data.items()
            if ticker in signals  # Only include stocks with signals
        }
        
        results = backtester.run(
            data=price_data,
            signals=signals,
            start_date=None,
            end_date=None
        )
        
        backtester.print_results()
        backtester.plot_results(save_path=f"{self.output_dir}/results/")
        backtester.export_trades(f"{self.output_dir}/results/trades.csv")
        
        return results
    
    def run_full_pipeline(self, n_clusters: int = 5, horizons: list = [5, 10]):
        """Run complete pipeline"""
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print(" "*15 + "ML TRADING SYSTEM PIPELINE")
        print("="*70)
        print(f"Tickers: {len(self.tickers)}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Output directory: {self.output_dir}")
        print(f"Started: {start_time}")
        print("="*70)
        
        try:
            # Step 1: Data Collection
            processed_data = self.step1_collect_data()
            
            # Step 2: Clustering
            cluster_results = self.step2_cluster_stocks(processed_data, n_clusters)
            
            # Step 3: Train Models
            self.step3_train_models(cluster_results, horizons)
            
            # Step 4: Generate Signals (FIXED)
            signals = self.step4_generate_signals(cluster_results, processed_data, 
                                                  horizon=horizons[0])
            
            # Step 5: Backtest
            backtest_results = self.step5_backtest(processed_data, signals)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print("\n" + "="*70)
            print(" "*20 + "PIPELINE COMPLETE!")
            print("="*70)
            print(f"\nExecution time: {duration}")
            print(f"Results saved to: {self.output_dir}")
            
            # Summary
            print("\n" + "="*70)
            print("EXECUTION SUMMARY:")
            print(f"  - Data collected: {len(processed_data)} stocks")
            print(f"  - Clusters formed: {len(np.unique(cluster_results['labels']))}")
            print(f"  - Models trained: {len(self.models)}")
            print(f"  - Signals generated: {len(signals)} stocks")
            
            if backtest_results and 'error' not in backtest_results:
                print(f"  - Total return: {backtest_results['total_return_pct']:.2f}%")
                print(f"  - CAGR: {backtest_results['cagr_pct']:.2f}%")
                print(f"  - Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}")
                print(f"  - Max drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
                print(f"  - Win rate: {backtest_results['win_rate']:.2f}%")
            print("="*70)
            
            return {
                'processed_data': processed_data,
                'clusters': cluster_results,
                'models': self.models,
                'signals': signals,
                'backtest': backtest_results,
                'execution_time': duration
            }
            
        except Exception as e:
            print(f"\n✗ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main execution"""
    
    # Get expanded stock universe (150+ stocks)
    tickers = get_expanded_stock_universe()
    
    print(f"Universe size: {len(tickers)} stocks")
    
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
        n_clusters=5,
        horizons=[5, 10]  # 5-day and 10-day predictions
    )
    
    if results:
        print("\n✓ Pipeline completed successfully!")
    else:
        print("\n✗ Pipeline failed!")


if __name__ == "__main__":
    main()