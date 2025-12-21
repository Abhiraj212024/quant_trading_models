"""
backtester.py - Comprehensive backtesting with IMPROVED equity visualization
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
    
    def can_open_position(self) -> bool:
        return len(self.positions) < self.config.max_positions
    
    def open_position(self, ticker: str, price: float, date: pd.Timestamp,
                     signal_strength: float = 1.0):
        if not self.can_open_position() or ticker in self.positions:
            return False
        
        available_capital = self.cash / max(1, (self.config.max_positions - len(self.positions)))
        position_value = available_capital * self.config.position_size * signal_strength
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
            'entry_date': date
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
            reason=reason
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
                current_price = prices.get(ticker)
                
                if current_price is None:
                    continue
                
                if signal == 1:
                    if ticker not in self.portfolio.positions:
                        self.portfolio.open_position(ticker, current_price, date, confidence)
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
    
    def export_trades(self, filepath: str):
        """Export trade history"""
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        trades_df.to_csv(filepath, index=False)
        print(f"✓ Trades exported to {filepath}")


if __name__ == "__main__":
    print("Enhanced backtester loaded with improved equity visualization")