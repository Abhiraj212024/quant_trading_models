"""
stochastic_models.py - Probabilistic models for trading signal confidence
"""
import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
import pymc as pm
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """Monte Carlo simulation for price paths and probability estimation"""
    
    def __init__(self, returns: np.ndarray, n_simulations: int = 10000):
        self.returns = returns
        self.n_simulations = n_simulations
        self.mu = np.mean(returns)
        self.sigma = np.std(returns)
        
    def simulate_paths(self, current_price: float, n_days: int) -> np.ndarray:
        """
        Simulate future price paths using Geometric Brownian Motion
        """
        dt = 1  # Daily timestep
        
        # Generate random returns
        random_returns = np.random.normal(
            self.mu * dt,
            self.sigma * np.sqrt(dt),
            size=(self.n_simulations, n_days)
        )
        
        # Calculate price paths
        price_paths = np.zeros((self.n_simulations, n_days + 1))
        price_paths[:, 0] = current_price
        
        for t in range(1, n_days + 1):
            price_paths[:, t] = price_paths[:, t-1] * (1 + random_returns[:, t-1])
        
        return price_paths
    
    def calculate_probabilities(self, current_price: float, n_days: int,
                                target_price: float = None) -> Dict:
        """Calculate various trading probabilities"""
        paths = self.simulate_paths(current_price, n_days)
        final_prices = paths[:, -1]
        
        results = {
            'prob_up': np.mean(final_prices > current_price),
            'prob_down': np.mean(final_prices < current_price),
            'expected_price': np.mean(final_prices),
            'median_price': np.median(final_prices),
            'price_std': np.std(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_25': np.percentile(final_prices, 25),
            'percentile_75': np.percentile(final_prices, 75),
            'percentile_95': np.percentile(final_prices, 95),
        }
        
        if target_price:
            results['prob_reach_target'] = np.mean(final_prices >= target_price)
            results['prob_stop_loss'] = np.mean(final_prices <= target_price * 0.95)
        
        # Value at Risk (VaR) and Conditional VaR (CVaR)
        returns_dist = (final_prices / current_price) - 1
        results['var_95'] = np.percentile(returns_dist, 5)
        results['cvar_95'] = np.mean(returns_dist[returns_dist <= results['var_95']])
        
        return results
    
    def simulate_with_jump_diffusion(self, current_price: float, n_days: int,
                                     jump_intensity: float = 0.1,
                                     jump_mean: float = -0.02,
                                     jump_std: float = 0.03) -> np.ndarray:
        """
        Merton Jump Diffusion Model - accounts for sudden market moves
        """
        dt = 1
        paths = np.zeros((self.n_simulations, n_days + 1))
        paths[:, 0] = current_price
        
        for t in range(1, n_days + 1):
            # Brownian motion component
            dW = np.random.normal(0, np.sqrt(dt), self.n_simulations)
            
            # Jump component (Poisson process)
            jumps = np.random.poisson(jump_intensity * dt, self.n_simulations)
            jump_sizes = np.random.normal(jump_mean, jump_std, self.n_simulations)
            
            # Price evolution
            drift = (self.mu - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * dW
            jump_contribution = jumps * jump_sizes
            
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion + jump_contribution)
        
        return paths


class GARCHModel:
    """GARCH model for volatility forecasting"""
    
    def __init__(self, returns: pd.Series):
        self.returns = returns * 100  # Scale for numerical stability
        self.model = None
        self.fitted = None
        
    def fit(self, p: int = 1, q: int = 1, dist: str = 'normal'):
        """
        Fit GARCH(p, q) model
        dist: 'normal', 't', 'skewt'
        """
        self.model = arch_model(
            self.returns,
            vol='Garch',
            p=p,
            q=q,
            dist=dist
        )
        self.fitted = self.model.fit(disp='off')
        return self.fitted
    
    def forecast_volatility(self, horizon: int = 5) -> Dict:
        """Forecast future volatility"""
        if self.fitted is None:
            self.fit()
        
        forecasts = self.fitted.forecast(horizon=horizon)
        variance_forecast = forecasts.variance.values[-1, :]
        
        return {
            'volatility_forecast': np.sqrt(variance_forecast) / 100,  # Unscale
            'mean_volatility': np.mean(np.sqrt(variance_forecast)) / 100,
            'volatility_trend': 'increasing' if variance_forecast[-1] > variance_forecast[0] else 'decreasing'
        }
    
    def calculate_dynamic_var(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Calculate Value at Risk with GARCH volatility"""
        if self.fitted is None:
            self.fit()
        
        forecast = self.fitted.forecast(horizon=horizon)
        vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) / 100
        
        # VaR calculation
        z_score = stats.norm.ppf(1 - confidence)
        var = z_score * vol_forecast * np.sqrt(horizon)
        
        return var


class BayesianModel:
    """Bayesian inference for probabilistic predictions"""
    
    def __init__(self, returns: np.ndarray):
        self.returns = returns
        self.trace = None
        self.model = None
        
    def fit_bayesian_regression(self, X: np.ndarray, y: np.ndarray, 
                                n_samples: int = 2000):
        """
        Fit Bayesian linear regression with uncertainty quantification
        """
        with pm.Model() as self.model:
            # Priors
            alpha = pm.Normal('alpha', mu=0, sigma=1)
            beta = pm.Normal('beta', mu=0, sigma=1, shape=X.shape[1])
            sigma = pm.HalfNormal('sigma', sigma=1)
            
            # Linear model
            mu = alpha + pm.math.dot(X, beta)
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
            
            # Inference
            self.trace = pm.sample(n_samples, tune=1000, return_inferencedata=True, 
                                  progressbar=False)
        
        return self.trace
    
    def predict_with_uncertainty(self, X_new: np.ndarray) -> Dict:
        """Make predictions with credible intervals"""
        if self.trace is None:
            raise ValueError("Model not fitted. Run fit_bayesian_regression first.")
        
        # Extract posterior samples
        alpha_samples = self.trace.posterior['alpha'].values.flatten()
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, X_new.shape[1])
        sigma_samples = self.trace.posterior['sigma'].values.flatten()
        
        # Predictions for each posterior sample
        predictions = []
        for i in range(len(alpha_samples)):
            pred = alpha_samples[i] + X_new @ beta_samples[i]
            pred += np.random.normal(0, sigma_samples[i], size=pred.shape)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean': np.mean(predictions, axis=0),
            'median': np.median(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'credible_interval_95': (
                np.percentile(predictions, 2.5, axis=0),
                np.percentile(predictions, 97.5, axis=0)
            ),
            'prob_positive': np.mean(predictions > 0, axis=0)
        }


class KalmanFilter:
    """Kalman Filter for dynamic beta and trend estimation"""
    
    def __init__(self, transition_covariance: float = 1e-5,
                 observation_covariance: float = 1e-2):
        self.Q = transition_covariance  # Process noise
        self.R = observation_covariance  # Measurement noise
        
    def estimate_dynamic_beta(self, returns: np.ndarray, 
                             market_returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate time-varying beta using Kalman filter
        """
        n = len(returns)
        
        # Initialize
        beta = np.zeros(n)
        P = np.zeros(n)  # Error covariance
        
        beta[0] = 1.0  # Initial beta
        P[0] = 1.0
        
        for t in range(1, n):
            # Prediction
            beta_pred = beta[t-1]
            P_pred = P[t-1] + self.Q
            
            # Update
            K = P_pred * market_returns[t] / (market_returns[t]**2 * P_pred + self.R)
            beta[t] = beta_pred + K * (returns[t] - beta_pred * market_returns[t])
            P[t] = (1 - K * market_returns[t]) * P_pred
        
        return beta, P


class EnsembleProbability:
    """Combine multiple probabilistic models for robust predictions"""
    
    def __init__(self, returns: pd.Series):
        self.returns = returns
        self.mc_simulator = MonteCarloSimulator(returns.values)
        self.garch_model = GARCHModel(returns)
        
    def get_comprehensive_probabilities(self, current_price: float, 
                                       horizon: int = 5,
                                       ml_prediction: float = None) -> Dict:
        """
        Combine multiple stochastic models for probability estimation
        """
        # Monte Carlo probabilities
        mc_results = self.mc_simulator.calculate_probabilities(
            current_price, horizon
        )
        
        # GARCH volatility forecast
        garch_results = self.garch_model.forecast_volatility(horizon)

        def logit(p):
            return np.log(p / (1-p + 1e-8))
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Combine with ML prediction if available
        if ml_prediction is not None:
            # Adjust probabilities based on ML model confidence
            ml_logit = logit(ml_prediction)
            mc_logit = logit(mc_results['prob_up'])
            ml_weight = np.tanh(abs(ml_logit))
            mc_weight = 1.0
            adjusted_prob_up = sigmoid(mc_weight * mc_logit + ml_weight * ml_logit)
            mc_results['ml_adjusted_prob_up'] = adjusted_prob_up
            mc_results['ml_confidence'] = ml_prediction
        kelly_prob = mc_results.get('ml_adjusted_prob_up', mc_results['prob_up'])
        # Risk metrics
        results = {
            **mc_results,
            **garch_results,
            'kelly_criterion': self._calculate_kelly(
                kelly_prob,
                mc_results['expected_price'] / current_price
            ),
            'sharpe_ratio_forecast': self._forecast_sharpe(
                mc_results['expected_price'] / current_price - 1,
                garch_results['mean_volatility']
            )
        }
        
        return results
    
    def _calculate_kelly(self, prob_win: float, payoff_ratio: float) -> float:
        """Calculate Kelly Criterion for position sizing"""
        if payoff_ratio <= 0:
            return 0
        kelly = (prob_win * payoff_ratio - (1 - prob_win)) / payoff_ratio
        return max(0, min(kelly, 0.25))  # Cap at 25% for safety
    
    def _forecast_sharpe(self, expected_return: float, volatility: float) -> float:
        """Forecast Sharpe ratio"""
        if volatility == 0:
            return 0
        return expected_return / volatility
    
    def generate_trading_signal(self, current_price: float, horizon: int,
                               ml_prediction: float, threshold: float = 0.6) -> Dict:
        """
        Generate trading signal with confidence scores
        """
        probs = self.get_comprehensive_probabilities(
            current_price, horizon, ml_prediction
        )
        
        # Combined probability
        combined_prob = probs.get('ml_adjusted_prob_up', probs['prob_up'])
        action = 'HOLD'
        if combined_prob > threshold:
            action = 'BUY'
        elif combined_prob < (1 - threshold):
            action = 'SELL'
        
        if probs.get('ml_confidence', probs['prob_up']) < 0.55:
            action = 'HOLD'  # Low confidence from ML model
        signal = {
            'action': action,
            'confidence': combined_prob if combined_prob > 0.5 else 1 - combined_prob,
            'probability_up': combined_prob,
            'expected_return': (probs['expected_price'] / current_price - 1) * 100,
            'risk_var_95': probs['var_95'] * 100,
            'position_size_kelly': probs['kelly_criterion'] * 100,
            'volatility_forecast': probs['mean_volatility'] * 100,
            'sharpe_forecast': probs['sharpe_ratio_forecast'],
            'price_targets': {
                'expected': probs['expected_price'],
                'pessimistic': probs['percentile_5'],
                'optimistic': probs['percentile_95']
            }
        }
        
        return signal


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate returns
    returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
    
    # Create ensemble
    ensemble = EnsembleProbability(returns)
    
    # Generate signal
    signal = ensemble.generate_trading_signal(
        current_price=100,
        horizon=5,
        ml_prediction=0.65
    )
    
    print("Trading Signal:")
    for key, value in signal.items():
        print(f"  {key}: {value}")