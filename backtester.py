"""
backtester.py - MODIFIED to support weighted position sizing + enhanced probability plots
CHANGES:
1. Portfolio.open_position() - now accepts position_size_pct parameter (line ~80)
2. Backtester.run() - reads position_size from signals (line ~220)
3. Added plot_probability_analysis() - new method (line ~450)
4. Added plot_model_performance() - new method (line ~550)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    position_size: float = 1.0
    max_positions: int = 10
    stop_loss: float = 0.05  # 5%
    take_profit: float = 0.15  # 15%
    risk_free_rate: float = 0.02


@dataclass
class Trade:
    """Individual trade record"""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    side: str
    pnl: float
    pnl_pct: float
    commission: float
    reason: str
    position_size_pct: float = 0.0  # CHANGE: Added to track position size


class Portfolio:
    """Portfolio manager"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def get_position_value(self, ticker: str, current_price: float) -> float:
        if ticker not in self.positions:
            return 0
        return self.positions[ticker]['shares'] * current_price
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        positions_value = sum(
            self.get_position_value(ticker, prices.get(ticker, 0))
            for ticker in self.positions
        )
        return self.cash + positions_value
    
    def can_open_position(self, prices: Dict[str, float]) -> bool:
        total_equity = self.get_total_value(prices)
        invested = sum(
            self.get_position_value(t, prices.get(t, 0))
            for t in self.positions
        )
        return invested / total_equity < 0.95

    
    # ==================== CHANGE 1: Modified to accept position_size_pct ====================
    def open_position(self, ticker: str, price: float, date: pd.Timestamp,
                     position_size_pct: float = None, prices: Dict[str, float] = None) -> bool:
        """
        Open a position with weighted sizing
        
        CHANGED: Now accepts position_size_pct parameter (0-100 scale)
        If None, uses default equal-weight allocation
        """
        if not self.can_open_position(prices) or ticker in self.positions:
            return False
        
        # CHANGE: Use provided position_size_pct if available
        if position_size_pct is not None and position_size_pct > 0:
            # position_size_pct is already scaled (0-8% typical range from main.py)
            # Convert to fraction of total portfolio value
            total_portfolio_value = self.cash + sum(
                pos['shares'] * prices.get(t, pos['entry_price'])
                for t, pos in self.positions.items()
            )


            position_value = total_portfolio_value * (position_size_pct / 100.0)
        else:
            # Default: equal weight allocation
            available_capital = self.cash / max(1, (self.config.max_positions - len(self.positions)))
            position_value = available_capital * self.config.position_size
        
        effective_price = price * (1 + self.config.slippage)
        shares = int(position_value / effective_price)
        
        if shares <= 0:
            return False
        
        total_cost = shares * effective_price
        commission = total_cost * self.config.commission
        
        if total_cost + commission > self.cash:
            return False
        
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': effective_price,
            'entry_date': date,
            'position_size_pct': position_size_pct or 0.0  # CHANGE: Store position size
        }
        self.cash -= (total_cost + commission)
        return True
    
    def close_position(self, ticker: str, price: float, date: pd.Timestamp, reason: str = 'signal'):
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        effective_price = price * (1 - self.config.slippage)
        proceeds = position['shares'] * effective_price
        commission = proceeds * self.config.commission
        cost_basis = position['shares'] * position['entry_price']
        
        pnl = proceeds - cost_basis - commission
        pnl_pct = (effective_price / position['entry_price'] - 1) * 100
        
        trade = Trade(
            ticker=ticker,
            entry_date=position['entry_date'],
            exit_date=date,
            entry_price=position['entry_price'],
            exit_price=effective_price,
            shares=position['shares'],
            side='long',
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            reason=reason,
            position_size_pct=position.get('position_size_pct', 0.0)  # CHANGE: Include in trade record
        )
        self.trades.append(trade)
        self.cash += proceeds - commission
        del self.positions[ticker]
    
    def check_stop_loss_take_profit(self, prices: Dict[str, float], date: pd.Timestamp):
        to_close = []
        for ticker, position in self.positions.items():
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            pnl_pct = (current_price / position['entry_price'] - 1)
            
            if pnl_pct <= -self.config.stop_loss:
                to_close.append((ticker, current_price, 'stop_loss'))
            elif pnl_pct >= self.config.take_profit:
                to_close.append((ticker, current_price, 'take_profit'))
        
        for ticker, price, reason in to_close:
            self.close_position(ticker, price, date, reason)


class Backtester:
    """Main backtesting engine with ENHANCED visualization"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.portfolio = Portfolio(self.config)
        self.results = None
        self.signals_history = []  # CHANGE: Store signal history for analysis
        
    def run(self, data: Dict[str, pd.DataFrame], 
            signals: Dict[str, pd.DataFrame],
            start_date: str = None, end_date: str = None) -> Dict:
        """Run backtest"""
        print("Running backtest...")
        
        # Get common dates
        all_dates = set.intersection(*[set(df.index) for df in data.values()])
        dates = sorted(list(all_dates))
        
        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]
        
        print(f"  Period: {dates[0].date()} to {dates[-1].date()}")
        print(f"  Trading days: {len(dates)}")
        print(f"  Stocks: {len(data)}")
        
        # Daily loop
        for i, date in enumerate(dates):
            if i % 50 == 0:
                print(f"  Processing day {i+1}/{len(dates)}...")
            
            prices = {ticker: df.loc[date, 'close'] 
                     for ticker, df in data.items() if date in df.index}
            
            self.portfolio.check_stop_loss_take_profit(prices, date)
            
            # Process signals
            for ticker, signal_df in signals.items():
                if ticker not in data or date not in signal_df.index:
                    continue
                
                signal_row = signal_df.loc[date]
                signal = signal_row.get('signal', 0)
                confidence = signal_row.get('confidence', 0.5)
                probability_up = signal_row.get('probability_up', 0.5)
                position_size = signal_row.get('position_size', None)  # CHANGE 2: Read position_size
                
                current_price = prices.get(ticker)
                
                if current_price is None:
                    continue
                
                # CHANGE: Store signal for later analysis
                self.signals_history.append({
                    'date': date,
                    'ticker': ticker,
                    'signal': signal, 
                    'confidence': confidence,
                    'probability_up': probability_up,
                    'position_size': position_size,
                    'price': current_price
                })
                
                if signal == 1:
                    if ticker not in self.portfolio.positions:
                        # CHANGE: Pass position_size to open_position
                        self.portfolio.open_position(ticker, current_price, date, position_size, prices)
                elif signal == -1:
                    if ticker in self.portfolio.positions:
                        self.portfolio.close_position(ticker, current_price, date, 'signal')
            
            # Record equity
            total_value = self.portfolio.get_total_value(prices)
            self.portfolio.equity_curve.append({
                'date': date,
                'equity': total_value,
                'cash': self.portfolio.cash,
                'positions_value': total_value - self.portfolio.cash,
                'num_positions': len(self.portfolio.positions)
            })
            
            if i > 0:
                prev_equity = self.portfolio.equity_curve[-2]['equity']
                daily_return = (total_value / prev_equity - 1)
                self.portfolio.daily_returns.append(daily_return)
        
        # Close remaining positions
        final_prices = {ticker: df.iloc[-1]['close'] for ticker, df in data.items()}
        final_date = dates[-1]
        for ticker in list(self.portfolio.positions.keys()):
            if ticker in final_prices:
                self.portfolio.close_position(ticker, final_prices[ticker], 
                                            final_date, 'end_of_backtest')
        
        self.results = self._calculate_metrics()
        print("\n✓ Backtest complete")
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        
        if len(trades_df) == 0:
            return {'error': 'No trades executed'}
        
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        final_equity = equity_df.iloc[-1]['equity']
        total_return = (final_equity / self.config.initial_capital - 1) * 100
        
        returns = np.array(self.portfolio.daily_returns)
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_df['equity'].values)
        
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        trades_df['holding_period'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        avg_holding_period = trades_df['holding_period'].mean()
        
        # Calculate CAGR
        years = len(equity_df) / 252
        cagr = (final_equity / self.config.initial_capital) ** (1/years) - 1 if years > 0 else 0
        
        metrics = {
            'total_return_pct': total_return,
            'cagr_pct': cagr * 100,
            'final_equity': final_equity,
            'total_trades': total_trades,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'max_drawdown_duration_days': max_drawdown_duration,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'avg_holding_period_days': avg_holding_period,
            'total_commission_paid': trades_df['commission'].sum(),
            'trades_per_month': total_trades / (len(equity_df) / 21),
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = returns - (self.config.risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
    
    def _calculate_sortino(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        if len(returns) == 0:
            return 0
        excess_returns = returns - (self.config.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)
    
    def _calculate_drawdown(self, equity: np.ndarray) -> Tuple[float, int]:
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = np.min(drawdown)
        
        is_drawdown = drawdown < 0
        dd_durations = []
        current_duration = 0
        
        for dd in is_drawdown:
            if dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    dd_durations.append(current_duration)
                current_duration = 0
        
        max_dd_duration = max(dd_durations) if dd_durations else 0
        return max_dd, max_dd_duration
    
    def print_results(self):
        """Print backtest results"""
        if self.results is None or 'error' in self.results:
            print("\n" + "="*60)
            print("ERROR: No trades executed or backtest failed")
            print("="*60)
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nPerformance:")
        print(f"  Total Return: {self.results['total_return_pct']:.2f}%")
        print(f"  CAGR: {self.results['cagr_pct']:.2f}%")
        print(f"  Final Equity: ${self.results['final_equity']:,.2f}")
        print(f"  Sharpe Ratio: {self.results['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {self.results['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {self.results['max_drawdown_pct']:.2f}%")
        print(f"  Max DD Duration: {self.results['max_drawdown_duration_days']:.0f} days")
        
        print(f"\nTrade Statistics:")
        print(f"  Total Trades: {self.results['total_trades']}")
        print(f"  Win Rate: {self.results['win_rate']:.2f}%")
        print(f"  Profit Factor: {self.results['profit_factor']:.2f}")
        print(f"  Avg Win: {self.results['avg_win_pct']:.2f}%")
        print(f"  Avg Loss: {self.results['avg_loss_pct']:.2f}%")
        print(f"  Avg Holding: {self.results['avg_holding_period_days']:.1f} days")
        print(f"  Trades/Month: {self.results['trades_per_month']:.1f}")
        
        print(f"\nCosts:")
        print(f"  Total Commission: ${self.results['total_commission_paid']:,.2f}")
        print("="*60)
    
    def plot_results(self, save_path: str = None):
        """ENHANCED visualization with prominent equity curve"""
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        
        if len(trades_df) == 0:
            print("No trades to plot")
            return
        
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. MAIN EQUITY CURVE (larger, more prominent)
        ax1 = fig.add_subplot(gs[0:2, :])
        ax1.plot(equity_df['date'], equity_df['equity'], linewidth=2, 
                label='Portfolio Value', color='#2E86AB')
        ax1.fill_between(equity_df['date'], equity_df['cash'], 
                        equity_df['equity'], alpha=0.3, label='Invested Capital',
                        color='#A23B72')
        
        # Add benchmark (buy and hold initial capital)
        initial = self.config.initial_capital
        ax1.axhline(y=initial, color='gray', linestyle='--', 
                   linewidth=1, label='Initial Capital', alpha=0.7)
        
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Drawdown
        ax2 = fig.add_subplot(gs[2, :])
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        ax2.fill_between(equity_df['date'], drawdown, 0, alpha=0.5, color='red')
        ax2.set_title('Drawdown Over Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Returns distribution
        ax3 = fig.add_subplot(gs[3, 0])
        returns = pd.Series(self.portfolio.daily_returns) * 100
        returns.hist(bins=50, ax=ax3, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(returns.mean(), color='r', linestyle='--', 
                   label=f'Mean: {returns.mean():.2f}%', linewidth=2)
        ax3.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Return (%)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade P&L distribution
        ax4 = fig.add_subplot(gs[3, 1])
        trades_df['pnl_pct'].hist(bins=40, ax=ax4, alpha=0.7, 
                                  color='green', edgecolor='black')
        ax4.axvline(0, color='black', linestyle='-', linewidth=1)
        ax4.axvline(trades_df['pnl_pct'].mean(), color='r', linestyle='--',
                   label=f'Mean: {trades_df["pnl_pct"].mean():.2f}%', linewidth=2)
        ax4.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('P&L (%)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Backtest Performance Summary', fontsize=18, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(f"{save_path}/backtest_results.png", dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}/backtest_results.png")
        
        plt.show()
    def plot_probability_analysis(self, save_path: str = None):
        """
        NEW METHOD: Analyze probability predictions vs actual outcomes
        """
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        signals_df = pd.DataFrame(self.signals_history)
        
        if len(trades_df) == 0 or len(signals_df) == 0:
            print("No data for probability analysis")
            return
        
        # Merge trades with their entry signals
        trades_with_signals = []
        for _, trade in trades_df.iterrows():
            matching_signal = signals_df[
                (signals_df['ticker'] == trade['ticker']) & 
                (signals_df['date'] == trade['entry_date'])
            ]
            if len(matching_signal) > 0:
                trades_with_signals.append({
                    'probability_up': matching_signal.iloc[0]['probability_up'],
                    'confidence': matching_signal.iloc[0]['confidence'],
                    'position_size': matching_signal.iloc[0]['position_size'],
                    'pnl_pct': trade['pnl_pct'],
                    'win': 1 if trade['pnl'] > 0 else 0
                })
        
        analysis_df = pd.DataFrame(trades_with_signals)
        
        if len(analysis_df) == 0:
            print("No matched trades for analysis")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Probability & Model Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Probability calibration curve
        ax = axes[0, 0]
        prob_bins = np.linspace(0, 1, 11)
        bin_centers = (prob_bins[:-1] + prob_bins[1:]) / 2
        
        actual_win_rates = []
        for i in range(len(prob_bins) - 1):
            mask = (analysis_df['probability_up'] >= prob_bins[i]) & \
                   (analysis_df['probability_up'] < prob_bins[i+1])
            if mask.sum() > 0:
                actual_win_rates.append(analysis_df[mask]['win'].mean())
            else:
                actual_win_rates.append(np.nan)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.5)
        ax.plot(bin_centers, actual_win_rates, 'o-', linewidth=2, markersize=8, 
                label='Actual Win Rate', color='#2E86AB')
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Actual Win Rate', fontsize=11)
        ax.set_title('Probability Calibration', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Confidence vs P&L
        ax = axes[0, 1]
        scatter = ax.scatter(analysis_df['confidence'], analysis_df['pnl_pct'], 
                           c=analysis_df['win'], cmap='RdYlGn', alpha=0.6, s=50)
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Confidence', fontsize=11)
        ax.set_ylabel('P&L (%)', fontsize=11)
        ax.set_title('Confidence vs Actual Returns', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Win')
        ax.grid(True, alpha=0.3)
        
        # 3. Position size distribution
        ax = axes[0, 2]
        analysis_df['position_size'].hist(bins=30, ax=ax, alpha=0.7, 
                                         color='purple', edgecolor='black')
        ax.axvline(analysis_df['position_size'].mean(), color='r', 
                  linestyle='--', linewidth=2, 
                  label=f'Mean: {analysis_df["position_size"].mean():.2f}%')
        ax.set_xlabel('Position Size (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Position Size Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Position size vs P&L
        ax = axes[1, 0]
        scatter = ax.scatter(analysis_df['position_size'], analysis_df['pnl_pct'],
                           c=analysis_df['probability_up'], cmap='viridis', 
                           alpha=0.6, s=50)
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax.set_xlabel('Position Size (%)', fontsize=11)
        ax.set_ylabel('P&L (%)', fontsize=11)
        ax.set_title('Position Sizing Effectiveness', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Prob Up')
        ax.grid(True, alpha=0.3)
        
        # 5. Win rate by probability bucket
        ax = axes[1, 1]
        prob_ranges = ['0-50%', '50-60%', '60-70%', '70-80%', '80-100%']
        win_rates = []
        counts = []
        
        for low, high in [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
            mask = (analysis_df['probability_up'] >= low) & \
                   (analysis_df['probability_up'] < high)
            if mask.sum() > 0:
                win_rates.append(analysis_df[mask]['win'].mean() * 100)
                counts.append(mask.sum())
            else:
                win_rates.append(0)
                counts.append(0)
        
        bars = ax.bar(prob_ranges, win_rates, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axhline(50, color='red', linestyle='--', linewidth=1, label='50% Baseline')
        ax.set_ylabel('Win Rate (%)', fontsize=11)
        ax.set_title('Win Rate by Probability Bucket', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add counts on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'n={count}', ha='center', va='bottom', fontsize=9)
        
        # 6. Expected vs Actual returns by probability bucket
        ax = axes[1, 2]
        avg_returns = []
        for low, high in [(0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]:
            mask = (analysis_df['probability_up'] >= low) & \
                   (analysis_df['probability_up'] < high)
            if mask.sum() > 0:
                avg_returns.append(analysis_df[mask]['pnl_pct'].mean())
            else:
                avg_returns.append(0)
        
        ax.bar(prob_ranges, avg_returns, alpha=0.7, color='green', edgecolor='black')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('Average Return (%)', fontsize=11)
        ax.set_title('Avg Return by Probability Bucket', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/probability_analysis.png", dpi=300, bbox_inches='tight')
            print(f"✓ Probability analysis saved to {save_path}/probability_analysis.png")
        
        plt.show()
        
        # Print statistics
        print("\n" + "="*60)
        print("PROBABILITY ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total trades analyzed: {len(analysis_df)}")
        print(f"Average predicted probability: {analysis_df['probability_up'].mean():.3f}")
        print(f"Actual win rate: {analysis_df['win'].mean():.3f}")
        print(f"Calibration error: {abs(analysis_df['probability_up'].mean() - analysis_df['win'].mean()):.3f}")
        print("="*60)
    
    # ==================== CHANGE 4: NEW METHOD - Model Performance ====================
    def plot_model_performance(self, save_path: str = None):
        """
        NEW METHOD: Detailed model performance metrics over time
        """
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        
        if len(trades_df) == 0:
            print("No trades for model performance analysis")
            return
        
        # Calculate rolling metrics
        trades_df = trades_df.sort_values('entry_date')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        trades_df['win'] = (trades_df['pnl'] > 0).astype(int)
        trades_df['rolling_win_rate'] = trades_df['win'].rolling(20, min_periods=1).mean() * 100
        trades_df['rolling_avg_pnl'] = trades_df['pnl_pct'].rolling(20, min_periods=1).mean()
        
        fig = plt.figure(figsize=(18, 14))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
        
        # 1. Cumulative P&L over time
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(trades_df['entry_date'], trades_df['cumulative_pnl'], 
                linewidth=2, color='#2E86AB')
        ax1.fill_between(trades_df['entry_date'], 0, trades_df['cumulative_pnl'],
                        alpha=0.3, color='#2E86AB')
        ax1.set_title('Cumulative P&L Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative P&L ($)', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Rolling win rate
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(trades_df['entry_date'], trades_df['rolling_win_rate'],
                linewidth=2, color='green')
        ax2.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
        ax2.fill_between(trades_df['entry_date'], 50, trades_df['rolling_win_rate'],
                        alpha=0.3, color='green')
        ax2.set_title('Rolling Win Rate (20-trade window)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Win Rate (%)', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling average P&L
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(trades_df['entry_date'], trades_df['rolling_avg_pnl'],
                linewidth=2, color='purple')
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.fill_between(trades_df['entry_date'], 0, trades_df['rolling_avg_pnl'],
                        alpha=0.3, color='purple')
        ax3.set_title('Rolling Avg P&L% (20-trade window)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Avg P&L (%)', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        # 4. Trade duration distribution
        ax4 = fig.add_subplot(gs[2, 0])
        trades_df['holding_period'].hist(bins=30, ax=ax4, alpha=0.7, 
                                        color='orange', edgecolor='black')
        ax4.axvline(trades_df['holding_period'].median(), color='r',
                   linestyle='--', linewidth=2,
                   label=f'Median: {trades_df["holding_period"].median():.1f} days')
        ax4.set_xlabel('Holding Period (days)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Exit reason breakdown
        ax5 = fig.add_subplot(gs[2, 1])
        reason_counts = trades_df['reason'].value_counts()
        colors = {'signal': '#2E86AB', 'stop_loss': 'red', 
                 'take_profit': 'green', 'end_of_backtest': 'gray'}
        ax5.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%',
               colors=[colors.get(r, 'blue') for r in reason_counts.index],
               startangle=90)
        ax5.set_title('Exit Reason Distribution', fontsize=12, fontweight='bold')
        
        # 6. Win/Loss by position size
        ax6 = fig.add_subplot(gs[3, 0])
        wins = trades_df[trades_df['win'] == 1]
        losses = trades_df[trades_df['win'] == 0]
        
        if len(wins) > 0:
            ax6.scatter(wins['position_size_pct'], wins['pnl_pct'],
                       alpha=0.6, s=50, color='green', label='Wins')
        if len(losses) > 0:
            ax6.scatter(losses['position_size_pct'], losses['pnl_pct'],
                       alpha=0.6, s=50, color='red', label='Losses')
        
        ax6.axhline(0, color='black', linestyle='-', linewidth=1)
        ax6.set_xlabel('Position Size (%)', fontsize=11)
        ax6.set_ylabel('P&L (%)', fontsize=11)
        ax6.set_title('Win/Loss by Position Size', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Monthly returns heatmap
        ax7 = fig.add_subplot(gs[3, 1])
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df['month'] = equity_df['date'].dt.to_period('M')
        monthly_returns = equity_df.groupby('month')['equity'].apply(
            lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100
        )
        
        if len(monthly_returns) > 0:
            monthly_returns.plot(kind='bar', ax=ax7, color='steelblue', alpha=0.7)
            ax7.axhline(0, color='black', linestyle='-', linewidth=1)
            ax7.set_xlabel('Month', fontsize=11)
            ax7.set_ylabel('Return (%)', fontsize=11)
            ax7.set_title('Monthly Returns', fontsize=12, fontweight='bold')
            ax7.grid(True, alpha=0.3, axis='y')
            plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        if save_path:
            plt.savefig(f"{save_path}/model_performance.png", dpi=300, bbox_inches='tight')
            print(f"✓ Model performance saved to {save_path}/model_performance.png")
        
        plt.show()
    
    def export_trades(self, filepath: str):
        """Export trade history"""
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        trades_df.to_csv(filepath, index=False)
        print(f"✓ Trades exported to {filepath}")


if __name__ == "__main__":
    print("Enhanced backtester loaded with weighted position sizing and probability analysis")