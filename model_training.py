"""
model_training.py - Ensemble ML models WITHOUT look-ahead bias
CRITICAL FIX: StandardScaler removed, proper train/test split with time ordering
TensorFlow version with complete implementation
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

AUTOTUNE = tf.data.AUTOTUNE


# ==================== tf.data Sequence Dataset ====================
def sequence_dataset(X, y, seq_len, batch_size, shuffle=False):
    """
    Create sequence dataset without look-ahead bias
    Each sequence uses only past data to predict the next point
    """
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    
    # Create sliding windows
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)
    
    # Flatten windows
    ds = ds.flat_map(
        lambda x, y: tf.data.Dataset.zip((
            x.batch(seq_len + 1),
            y.batch(seq_len + 1)
        ))
    )
    
    # Split into input (first seq_len) and target (last point)
    ds = ds.map(
        lambda x, y: (x[:-1], y[-1]),
        num_parallel_calls=AUTOTUNE
    )
    
    if shuffle:
        ds = ds.shuffle(2048)
    
    return ds.batch(batch_size).prefetch(AUTOTUNE)


# ==================== Attention Layer ====================
class TemporalAttention(layers.Layer):
    """Attention mechanism for temporal data"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = layers.Dense(1)

    def call(self, x):
        scores = self.score(x)  # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)  # softmax over time
        context = tf.reduce_sum(weights * x, axis=1)  # weighted sum
        return context


# ==================== CNN-LSTM Model ====================
def build_cnn_lstm(input_dim, seq_len, hidden_dim=128, num_layers=2, dropout=0.2):
    """
    CNN-LSTM model for sequence prediction
    
    Args:
        input_dim: Number of input features
        seq_len: Sequence length (lookback window)
        hidden_dim: Hidden dimension size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    inputs = keras.Input(shape=(seq_len, input_dim))
    
    # Convolutional layers for feature extraction
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    
    # LSTM layers for temporal modeling
    for i in range(num_layers):
        x = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=dropout if num_layers > 1 else 0.0,
            recurrent_dropout=0.0
        )(x)
    
    # Attention mechanism
    context = TemporalAttention(hidden_dim)(x)
    
    # Output layers
    x = layers.Dense(64, activation="relu")(context)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compile with gradient clipping
    optimizer = keras.optimizers.Adam(1e-3, clipnorm=1.0)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# ==================== Positional Encoding ====================
class PositionalEncoding(layers.Layer):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = np.arange(max_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        self.pe = tf.constant(pe[None], dtype=tf.float32)

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]


# ==================== Transformer Model ====================
def build_transformer(input_dim, seq_len, d_model=128, nhead=8, num_layers=3, dropout=0.1):
    """
    Transformer model for sequence prediction
    
    Args:
        input_dim: Number of input features
        seq_len: Sequence length
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dropout: Dropout rate
    """
    inputs = keras.Input(shape=(seq_len, input_dim))
    
    # Project input to model dimension
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)
    
    # Transformer encoder layers
    for _ in range(num_layers):
        # Multi-head attention
        attn = layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model // nhead,
            dropout=dropout
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-5)(x + attn)
        
        # Feed-forward network
        ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation="relu"),
            layers.Dense(d_model)
        ])
        x = layers.LayerNormalization(epsilon=1e-5)(x + ffn(x))
    
    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    # Compile
    optimizer = keras.optimizers.Adam(1e-3, clipnorm=1.0)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# ==================== Ensemble Model ====================
class EnsembleModel:
    """
    Ensemble model WITHOUT look-ahead bias
    
    Key features:
    - No StandardScaler (removed to prevent look-ahead)
    - Proper time-based train/val split
    - Features normalized using only training data statistics
    - All models trained independently
    """
    
    def __init__(self, cluster_id: int, horizon: int = 5):
        """
        Initialize ensemble model
        
        Args:
            cluster_id: ID of the stock cluster
            horizon: Prediction horizon in days
        """
        self.cluster_id = cluster_id
        self.horizon = horizon
        self.models = {}
        self.feature_names = None
        self.sequence_length = 60
        
        # Store normalization parameters from training data only
        self.feature_means = None
        self.feature_stds = None
        
    def prepare_data(self, cluster_data: Dict[str, pd.DataFrame], train_ratio=0.7):
        """
        Prepare data with STRICT time-based split - NO LOOK-AHEAD BIAS
        
        Args:
            cluster_data: Dictionary of {ticker: DataFrame}
            train_ratio: Ratio of data to use for training
            
        Returns:
            X_train, X_val, y_train, y_val
        """
        all_data = []
        target_col = f"target_direction_{self.horizon}d"
        
        # Exclude columns that shouldn't be features
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
        exclude_patterns = ['target_']  # Exclude all target variables
        
        print(f"  Preparing data for {len(cluster_data)} stocks...")
        
        for ticker, df in cluster_data.items():
            # Only use rows where target is available (not NaN)
            df_valid = df.dropna(subset=[target_col]).copy()
            
            if len(df_valid) < self.sequence_length + 50:
                continue
            
            # Select features (only use past data, no targets as features)
            if self.feature_names is None:
                self.feature_names = [
                    c for c in df_valid.columns
                    if c not in exclude_cols and not any(p in c for p in exclude_patterns)
                ]
            
            # Ensure all features exist and have no NaN
            df_valid = df_valid[self.feature_names + [target_col]].dropna()
            
            if len(df_valid) < self.sequence_length + 50:
                continue
            
            all_data.append(df_valid)
        
        if not all_data:
            raise ValueError("No valid data available for training")
        
        # Concatenate all data and sort by time
        combined_df = pd.concat(all_data, axis=0).sort_index()
        
        X = combined_df[self.feature_names].values
        y = combined_df[target_col].values
        
        # TIME-BASED SPLIT (critical for no look-ahead bias)
        # Earlier data for training, later data for validation
        split_idx = int(len(X) * train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Normalize using ONLY training data statistics
        print(f"  Computing normalization from training data only...")
        self.feature_means = np.nanmean(X_train, axis=0)
        self.feature_stds = np.nanstd(X_train, axis=0)
        
        # Avoid division by zero
        self.feature_stds = np.where(self.feature_stds == 0, 1, self.feature_stds)
        
        # Apply normalization
        X_train = (X_train - self.feature_means) / self.feature_stds
        X_val = (X_val - self.feature_means) / self.feature_stds
        
        # Replace any remaining NaN/inf values
        X_train = np.nan_to_num(X_train, nan=0, posinf=3, neginf=-3)
        X_val = np.nan_to_num(X_val, nan=0, posinf=3, neginf=-3)
        
        print(f"  Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Target balance - Train: {y_train.mean():.2%}, Val: {y_val.mean():.2%}")
        
        return X_train, X_val, y_train, y_val
    
    def _normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features using training statistics
        
        Args:
            X: Input features
            
        Returns:
            Normalized features
        """
        if self.feature_means is None or self.feature_stds is None:
            return X
        
        X_norm = (X - self.feature_means) / self.feature_stds
        return np.nan_to_num(X_norm, nan=0, posinf=3, neginf=-3)
    
    def _train_model(self, model, X_train, y_train, X_val, y_val):
        """
        Train deep learning model
        
        Args:
            model: Keras model
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Dictionary of validation metrics
        """
        # Create sequence datasets
        train_ds = sequence_dataset(X_train, y_train, self.sequence_length, 32, shuffle=True)
        val_ds = sequence_dataset(X_val, y_val, self.sequence_length, 32)
        
        # Train model
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=30,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    patience=7, 
                    restore_best_weights=True,
                    monitor='val_loss'
                ),
                keras.callbacks.ReduceLROnPlateau(
                    patience=3, 
                    factor=0.5,
                    monitor='val_loss'
                )
            ],
            verbose=0
        )
        
        # Compute validation metrics
        preds, labels = [], []
        for xb, yb in val_ds:
            p = model.predict(xb, verbose=0).ravel()
            preds.extend(p > 0.5)
            labels.extend(yb.numpy())
        
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0)
        }
        
        return metrics
    
    def train_cnn_lstm(self, X_train, X_val, y_train, y_val):
        """Train CNN-LSTM model"""
        print("  Training CNN-LSTM...")
        model = build_cnn_lstm(X_train.shape[1], self.sequence_length)
        metrics = self._train_model(model, X_train, y_train, X_val, y_val)
        self.models["cnn_lstm"] = model
        print(f"    Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        return metrics
    
    def train_transformer(self, X_train, X_val, y_train, y_val):
        """Train Transformer model"""
        print("  Training Transformer...")
        model = build_transformer(X_train.shape[1], self.sequence_length)
        metrics = self._train_model(model, X_train, y_train, X_val, y_val)
        self.models["transformer"] = model
        print(f"    Accuracy: {metrics['accuracy']:.3f}, F1: {metrics['f1']:.3f}")
        return metrics
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        """Train LightGBM model"""
        print("  Training LightGBM...")
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbose": -1
        }
        
        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)
        
        self.models["lightgbm"] = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        
        # Calculate validation metrics
        val_preds = self.models["lightgbm"].predict(X_val)
        val_preds_binary = (val_preds > 0.5).astype(int)
        
        acc = accuracy_score(y_val, val_preds_binary)
        f1 = f1_score(y_val, val_preds_binary, zero_division=0)
        
        print(f"    Accuracy: {acc:.3f}, F1: {f1:.3f}")
        
        return {"accuracy": acc, "f1": f1}
    
    def predict(self, X, weights=None):
        """
        Make predictions using ensemble
        
        Args:
            X: Input features
            weights: Dictionary of model weights
            
        Returns:
            Ensemble predictions
        """
        if weights is None:
            weights = {
                "cnn_lstm": 0.35,
                "transformer": 0.35,
                "lightgbm": 0.30
            }
        
        # Normalize using training statistics
        X = self._normalize_features(X)
        
        preds = {}
        
        # Deep learning models (require sequences)
        for name in ["cnn_lstm", "transformer"]:
            if name in self.models:
                ds = sequence_dataset(X, np.zeros(len(X)), self.sequence_length, 32)
                preds[name] = np.concatenate([
                    self.models[name].predict(xb, verbose=0).ravel()
                    for xb, _ in ds
                ])
        
        # LightGBM (uses raw features)
        if "lightgbm" in self.models:
            preds["lightgbm"] = self.models["lightgbm"].predict(X)
        
        # Ensemble predictions
        if not preds:
            return np.zeros(len(X))
        
        # Weighted average
        ensemble_pred = sum(preds[k] * weights.get(k, 0) for k in preds if k in weights)
        
        return ensemble_pred
    
    def save(self, path):
        """
        Save model to disk
        
        Args:
            path: Base path for saving (without extension)
        """
        import pickle
        import os
        
        # Create directory if needed
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save Keras models
        for k in ["cnn_lstm", "transformer"]:
            if k in self.models:
                self.models[k].save(f"{path}_{k}.keras")
        
        # Save LightGBM model
        if "lightgbm" in self.models:
            self.models["lightgbm"].save_model(f"{path}_lightgbm.txt")
        
        # Save metadata and normalization parameters
        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({
                "cluster_id": self.cluster_id,
                "horizon": self.horizon,
                "feature_names": self.feature_names,
                "sequence_length": self.sequence_length,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds
            }, f)
        
        print(f"    Model saved to {path}")
    
    def load(self, path):
        """
        Load model from disk
        
        Args:
            path: Base path for loading (without extension)
        """
        import pickle
        import os
        
        # Load metadata
        with open(f"{path}_meta.pkl", "rb") as f:
            meta = pickle.load(f)
            self.cluster_id = meta["cluster_id"]
            self.horizon = meta["horizon"]
            self.feature_names = meta["feature_names"]
            self.sequence_length = meta["sequence_length"]
            self.feature_means = meta["feature_means"]
            self.feature_stds = meta["feature_stds"]
        
        # Load Keras models
        for k in ["cnn_lstm", "transformer"]:
            model_path = f"{path}_{k}.keras"
            if os.path.exists(model_path):
                self.models[k] = keras.models.load_model(
                    model_path,
                    custom_objects={"TemporalAttention": TemporalAttention}
                )
        
        # Load LightGBM model
        lgb_path = f"{path}_lightgbm.txt"
        if os.path.exists(lgb_path):
            self.models["lightgbm"] = lgb.Booster(model_file=lgb_path)
        
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 70)
    print("Model Training Module - NO LOOK-AHEAD BIAS")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ No StandardScaler (manual normalization)")
    print("  ✓ Time-based train/val split")
    print("  ✓ Normalization using training data only")
    print("  ✓ CNN-LSTM + Transformer + LightGBM ensemble")
    print("  ✓ Proper sequence construction")
    print("=" * 70)