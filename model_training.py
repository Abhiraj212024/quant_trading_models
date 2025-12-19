"""
ml_models.py - Ensemble of ML models for stock prediction
TensorFlow version with PyTorch parity + RAM efficiency
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

AUTOTUNE = tf.data.AUTOTUNE


# ==================== tf.data Sequence Dataset ====================
def sequence_dataset(X, y, seq_len, batch_size, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))

    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)

    ds = ds.flat_map(
        lambda x, y: tf.data.Dataset.zip((
            x.batch(seq_len + 1),
            y.batch(seq_len + 1)
        ))
    )

    ds = ds.map(
        lambda x, y: (x[:-1], y[-1]),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        ds = ds.shuffle(2048)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)



# ==================== Attention Layer (Parity Correct) ====================
class TemporalAttention(layers.Layer):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = layers.Dense(1)

    def call(self, x):
        scores = self.score(x)                         # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)        # softmax over time
        context = tf.reduce_sum(weights * x, axis=1)   # weighted sum
        return context


# ==================== CNN-LSTM Model ====================
def build_cnn_lstm(input_dim, seq_len, hidden_dim=128, num_layers=2, dropout=0.2):
    inputs = keras.Input(shape=(seq_len, input_dim))

    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)

    for i in range(num_layers):
        x = layers.LSTM(
            hidden_dim,
            return_sequences=True,
            dropout=dropout if num_layers > 1 else 0.0,
            recurrent_dropout=0.0
        )(x)

    context = TemporalAttention(hidden_dim)(x)

    x = layers.Dense(64, activation="relu")(context)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    optimizer = keras.optimizers.Adam(1e-3, clipnorm=1.0)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==================== Positional Encoding ====================
class PositionalEncoding(layers.Layer):
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
    inputs = keras.Input(shape=(seq_len, input_dim))
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(d_model)(x)

    for _ in range(num_layers):
        attn = layers.MultiHeadAttention(
            num_heads=nhead,
            key_dim=d_model // nhead,
            dropout=dropout
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-5)(x + attn)

        ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation="relu"),
            layers.Dense(d_model)
        ])
        x = layers.LayerNormalization(epsilon=1e-5)(x + ffn(x))

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

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
    def __init__(self, cluster_id: int, horizon: int = 5):
        self.cluster_id = cluster_id
        self.horizon = horizon
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.sequence_length = 60

    def prepare_data(self, cluster_data: Dict[str, pd.DataFrame], train_ratio=0.7):
        all_X, all_y = [], []
        target_col = f"target_direction_{self.horizon}d"

        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
        exclude_patterns = ['target_', 'day_of_', 'month', 'quarter', 'is_']

        for _, df in cluster_data.items():
            df = df.dropna(subset=[target_col])
            if len(df) < self.sequence_length + 20:
                continue

            if self.feature_names is None:
                self.feature_names = [
                    c for c in df.columns
                    if c not in exclude_cols and not any(p in c for p in exclude_patterns)
                ]

            all_X.append(df[self.feature_names].values)
            all_y.append(df[target_col].values)

        X = np.vstack(all_X)
        y = np.hstack(all_y)

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        self.scalers['features'] = StandardScaler()
        X = self.scalers['features'].fit_transform(X)

        split = int(len(X) * train_ratio)
        return X[:split], X[split:], y[:split], y[split:]

    def _train_model(self, model, X_train, y_train, X_val, y_val):
        train_ds = sequence_dataset(
            X_train, y_train, self.sequence_length, 32, shuffle=True
        )
        val_ds = sequence_dataset(
            X_val, y_val, self.sequence_length, 32
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ],
            verbose=1
        )

        # Compute parity metrics
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
        model = build_cnn_lstm(X_train.shape[1], self.sequence_length)
        metrics = self._train_model(model, X_train, y_train, X_val, y_val)
        self.models["cnn_lstm"] = model
        print("✓ CNN-LSTM metrics:", metrics)

    def train_transformer(self, X_train, X_val, y_train, y_val):
        model = build_transformer(X_train.shape[1], self.sequence_length)
        metrics = self._train_model(model, X_train, y_train, X_val, y_val)
        self.models["transformer"] = model
        print("✓ Transformer metrics:", metrics)

    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1
        }

        self.models["lightgbm"] = lgb.train(
            params,
            lgb.Dataset(X_train, y_train),
            valid_sets=[lgb.Dataset(X_val, y_val)],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50)]
        )

    def predict(self, X, weights=None):
        if weights is None:
            weights = {"cnn_lstm": 0.35, "transformer": 0.35, "lightgbm": 0.30}

        X = self.scalers["features"].transform(X)
        preds = {}

        for name in ["cnn_lstm", "transformer"]:
            if name in self.models:
                ds = sequence_dataset(X, np.zeros(len(X)), self.sequence_length, 32)
                preds[name] = np.concatenate([
                    self.models[name].predict(xb, verbose=0).ravel()
                    for xb, _ in ds
                ])

        if "lightgbm" in self.models:
            preds["lightgbm"] = self.models["lightgbm"].predict(X)

        return sum(preds[k] * weights[k] for k in preds)

    def save(self, path):
        import pickle
        for k in ["cnn_lstm", "transformer"]:
            if k in self.models:
                self.models[k].save(f"{path}_{k}.keras")

        if "lightgbm" in self.models:
            self.models["lightgbm"].save_model(f"{path}_lightgbm.txt")

        with open(f"{path}_meta.pkl", "wb") as f:
            pickle.dump({
                "cluster_id": self.cluster_id,
                "horizon": self.horizon,
                "feature_names": self.feature_names,
                "sequence_length": self.sequence_length,
                "scalers": self.scalers
            }, f)


if __name__ == "__main__":
    print("Parity-correct TensorFlow ensemble loaded.")
