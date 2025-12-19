"""
backtester.py - Comprehensive backtesting framework with transaction costs
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    initial_capital: float = 100000
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    position_size: float = 1.0  # Full allocation
    max_positions: int = 10
    stop_loss: float = 0.05  # 5%
    take_profit: float = 0.15  # 15%
    risk_free_rate: float = 0.02  # 2% annual


@dataclass
class Trade:
    """Individual trade record"""
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float
    reason: str  # 'signal', 'stop_loss', 'take_profit', 'timeout'


class Portfolio:
    """Portfolio manager for backtesting"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}  # {ticker: {'shares': int, 'entry_price': float, 'entry_date': date}}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
    def get_position_value(self, ticker: str, current_price: float) -> float:
        """Get current value of position"""
        if ticker not in self.positions:
            return 0
        return self.positions[ticker]['shares'] * current_price
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Get total portfolio value"""
        positions_value = sum(
            self.get_position_value(ticker, prices.get(ticker, 0))
            for ticker in self.positions
        )
        return self.cash + positions_value
    
    def can_open_position(self) -> bool:
        """Check if we can open new positions"""
        return len(self.positions) < self.config.max_positions
    
    def open_position(self, ticker: str, price: float, date: pd.Timestamp,
                     signal_strength: float = 1.0):
        """Open a new position"""
        if not self.can_open_position() or ticker in self.positions:
            return False
        
        # Calculate position size
        available_capital = self.cash / (self.config.max_positions - len(self.positions))
        position_value = available_capital * self.config.position_size * signal_strength
        
        # Account for slippage
        effective_price = price * (1 + self.config.slippage)
        shares = int(position_value / effective_price)
        
        if shares <= 0:
            return False
        
        total_cost = shares * effective_price
        commission = total_cost * self.config.commission
        
        if total_cost + commission > self.cash:
            return False
        
        # Execute trade
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': effective_price,
            'entry_date': date
        }
        self.cash -= (total_cost + commission)
        
        return True
    
    def close_position(self, ticker: str, price: float, date: pd.Timestamp,
                      reason: str = 'signal'):
        """Close an existing position"""
        if ticker not in self.positions:
            return
        
        position = self.positions[ticker]
        
        # Account for slippage
        effective_price = price * (1 - self.config.slippage)
        
        # Calculate P&L
        proceeds = position['shares'] * effective_price
        commission = proceeds * self.config.commission
        cost_basis = position['shares'] * position['entry_price']
        
        pnl = proceeds - cost_basis - commission
        pnl_pct = (effective_price / position['entry_price'] - 1) * 100
        
        # Record trade
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
        
        # Update cash
        self.cash += proceeds - commission
        
        # Remove position
        del self.positions[ticker]
    
    def check_stop_loss_take_profit(self, prices: Dict[str, float], 
                                    date: pd.Timestamp):
        """Check and execute stop loss and take profit orders"""
        to_close = []
        
        for ticker, position in self.positions.items():
            if ticker not in prices:
                continue
            
            current_price = prices[ticker]
            entry_price = position['entry_price']
            pnl_pct = (current_price / entry_price - 1)
            
            # Stop loss
            if pnl_pct <= -self.config.stop_loss:
                to_close.append((ticker, current_price, 'stop_loss'))
            # Take profit
            elif pnl_pct >= self.config.take_profit:
                to_close.append((ticker, current_price, 'take_profit'))
        
        for ticker, price, reason in to_close:
            self.close_position(ticker, price, date, reason)


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.portfolio = Portfolio(self.config)
        self.results = None
        
    def run(self, data: Dict[str, pd.DataFrame], 
            signals: Dict[str, pd.DataFrame],
            start_date: str = None, end_date: str = None) -> Dict:
        """
        Run backtest
        
        data: {ticker: DataFrame with OHLCV}
        signals: {ticker: DataFrame with 'signal' column (-1, 0, 1) and 'confidence'}
        """
        print("Running backtest...")
        
        # Get common dates
        all_dates = set.intersection(*[set(df.index) for df in data.values()])
        dates = sorted(list(all_dates))
        
        if start_date:
            dates = [d for d in dates if d >= pd.Timestamp(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.Timestamp(end_date)]
        
        print(f"  Backtesting {len(dates)} days")
        print(f"  Tracking {len(data)} tickers")
        
        # Daily loop
        for i, date in enumerate(dates):
            # Get current prices
            prices = {ticker: df.loc[date, 'close'] 
                     for ticker, df in data.items() if date in df.index}
            
            # Check stop loss / take profit
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
                
                # Execute signals
                if signal == 1:  # Buy signal
                    if ticker not in self.portfolio.positions:
                        self.portfolio.open_position(ticker, current_price, date, confidence)
                
                elif signal == -1:  # Sell signal
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
            
            # Calculate daily return
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
        
        # Calculate metrics
        self.results = self._calculate_metrics()
        print("\n✓ Backtest complete")
        
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        
        if len(trades_df) == 0:
            return {'error': 'No trades executed'}
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = trades_df[trades_df['pnl'] > 0]
        losing_trades = trades_df[trades_df['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Returns
        final_equity = equity_df.iloc[-1]['equity']
        total_return = (final_equity / self.config.initial_capital - 1) * 100
        
        # Risk metrics
        returns = np.array(self.portfolio.daily_returns)
        sharpe_ratio = self._calculate_sharpe(returns)
        sortino_ratio = self._calculate_sortino(returns)
        max_drawdown, max_drawdown_duration = self._calculate_drawdown(equity_df['equity'].values)
        
        # Trade metrics
        avg_win = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
        
        # Holding period
        trades_df['holding_period'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
        avg_holding_period = trades_df['holding_period'].mean()
        
        metrics = {
            'total_return_pct': total_return,
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
            'trades_per_month': total_trades / (len(equity_df) / 21),  # Assuming 21 trading days/month
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        excess_returns = returns - (self.config.risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(returns)
    
    def _calculate_sortino(self, returns: np.ndarray, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0
        excess_returns = returns - (self.config.risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0
        return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)
    
    def _calculate_drawdown(self, equity: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration"""
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        max_dd = np.min(drawdown)
        
        # Calculate duration
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
        if self.results is None:
            print("Run backtest first")
            return
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)

        if not self.results:
            print(f"Error: no trades executed.")
            return
        
        print(f"\nPerformance:")
        print(f"  Total Return: {self.results['total_return_pct']:.2f}%")
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
        print(f"  Avg Holding Period: {self.results['avg_holding_period_days']:.1f} days")
        print(f"  Trades/Month: {self.results['trades_per_month']:.1f}")
        
        print(f"\nCosts:")
        print(f"  Total Commission: ${self.results['total_commission_paid']:,.2f}")
        
        print("="*60)
    
    def plot_results(self, save_path: str = None):
        """Visualize backtest results"""
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Equity curve
        ax = axes[0, 0]
        ax.plot(equity_df['date'], equity_df['equity'], label='Portfolio Value')
        ax.fill_between(equity_df['date'], equity_df['cash'], 
                        equity_df['equity'], alpha=0.3, label='Positions')
        ax.set_title('Equity Curve')
        ax.set_ylabel('Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Drawdown
        ax = axes[0, 1]
        running_max = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - running_max) / running_max * 100
        ax.fill_between(equity_df['date'], drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # Returns distribution
        ax = axes[1, 0]
        returns = pd.Series(self.portfolio.daily_returns)
        returns.hist(bins=50, ax=ax, alpha=0.7)
        ax.axvline(returns.mean(), color='r', linestyle='--', label=f'Mean: {returns.mean()*100:.2f}%')
        ax.set_title('Daily Returns Distribution')
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Trade P&L distribution
        ax = axes[1, 1]
        trades_df['pnl_pct'].hist(bins=30, ax=ax, alpha=0.7)
        ax.axvline(0, color='black', linestyle='-', linewidth=1)
        ax.set_title('Trade P&L Distribution')
        ax.set_xlabel('P&L (%)')
        ax.set_ylabel('Frequency')
        
        # Cumulative P&L by ticker
        ax = axes[2, 0]
        ticker_pnl = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False).head(10)
        ticker_pnl.plot(kind='bar', ax=ax)
        ax.set_title('Top 10 Stocks by P&L')
        ax.set_ylabel('Total P&L ($)')
        ax.set_xlabel('Ticker')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Trade outcomes over time
        ax = axes[2, 1]
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        ax.plot(trades_df['exit_date'], trades_df['cumulative_pnl'])
        ax.set_title('Cumulative Trade P&L')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/backtest_results.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_trades(self, filepath: str):
        """Export trade history to CSV"""
        trades_df = pd.DataFrame([vars(t) for t in self.portfolio.trades])
        trades_df.to_csv(filepath, index=False)
        print(f"✓ Trades exported to {filepath}")


if __name__ == "__main__":
    print("Backtester loaded. Use Backtester class to run backtests.")