"""
ml_models.py - Ensemble of ML models for stock prediction
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# ==================== PyTorch Dataset ====================
class StockDataset(Dataset):
    def __init__(self, X, y, sequence_length=60):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length]
        )


# ==================== CNN-LSTM Model ====================
class CNNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super(CNNLSTM, self).__init__()
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout_cnn = nn.Dropout(dropout)
        
        # LSTM layers for temporal patterns
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout_fc = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # CNN: (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout_cnn(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Back to (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected
        x = self.relu(self.fc1(context))
        x = self.dropout_fc(x)
        x = self.sigmoid(self.fc2(x))
        
        return x.squeeze()


# ==================== Transformer Model ====================
class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, dropout=0.1):
        super(StockTransformer, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc1 = nn.Linear(d_model, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        
        return x.squeeze()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== Model Trainer ====================
class ModelTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item()
                
                all_preds.extend((outputs > 0.5).cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0)
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            self.scheduler.step(val_metrics['loss'])
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}")
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        return self.model


# ==================== Ensemble Model ====================
class EnsembleModel:
    def __init__(self, cluster_id: int, horizon: int = 5):
        self.cluster_id = cluster_id
        self.horizon = horizon
        self.models = {}
        self.scalers = {}
        self.feature_names = None
        self.sequence_length = 60
        
    def prepare_data(self, cluster_data: Dict[str, pd.DataFrame], 
                     train_ratio: float = 0.7) -> Tuple:
        """Prepare training and validation data from cluster"""
        all_X = []
        all_y = []
        
        target_col = f'target_direction_{self.horizon}d'
        
        # Feature selection (exclude target columns and temporal features)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits']
        exclude_patterns = ['target_', 'day_of_', 'month', 'quarter', 'is_']
        
        for ticker, df in cluster_data.items():
            df = df.dropna(subset=[target_col])
            
            if len(df) < self.sequence_length + 20:
                continue
            
            # Select features
            if self.feature_names is None:
                feature_cols = [col for col in df.columns 
                               if col not in exclude_cols 
                               and not any(pattern in col for pattern in exclude_patterns)]
                self.feature_names = feature_cols
            
            X = df[self.feature_names].values
            y = df[target_col].values
            
            all_X.append(X)
            all_y.append(y)
        
        # Concatenate all stocks
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        # Remove any remaining NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]
        
        # Scale features
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        
        # Split
        split_idx = int(len(X_scaled) * train_ratio)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"\nCluster {self.cluster_id} - Data prepared:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Positive class ratio: {y_train.mean():.2%}")
        
        return X_train, X_val, y_train, y_val
    
    def train_cnn_lstm(self, X_train, X_val, y_train, y_val):
        """Train CNN-LSTM model"""
        print("\nTraining CNN-LSTM...")
        
        train_dataset = StockDataset(X_train, y_train, self.sequence_length)
        val_dataset = StockDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = CNNLSTM(input_dim=X_train.shape[1], hidden_dim=128, num_layers=2)
        trainer = ModelTrainer(model)
        trained_model = trainer.train(train_loader, val_loader, epochs=50)
        
        self.models['cnn_lstm'] = trained_model
        print("✓ CNN-LSTM trained")
    
    def train_transformer(self, X_train, X_val, y_train, y_val):
        """Train Transformer model"""
        print("\nTraining Transformer...")
        
        train_dataset = StockDataset(X_train, y_train, self.sequence_length)
        val_dataset = StockDataset(X_val, y_val, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        model = StockTransformer(input_dim=X_train.shape[1], d_model=128, nhead=8, num_layers=3)
        trainer = ModelTrainer(model)
        trained_model = trainer.train(train_loader, val_loader, epochs=50)
        
        self.models['transformer'] = trained_model
        print("✓ Transformer trained")
    
    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        """Train LightGBM model"""
        print("\nTraining LightGBM...")
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=100)]
        )
        
        self.models['lightgbm'] = model
        print("✓ LightGBM trained")
    
    def predict(self, X: np.ndarray, weights: Dict[str, float] = None) -> np.ndarray:
        """Ensemble prediction"""
        if weights is None:
            weights = {'cnn_lstm': 0.35, 'transformer': 0.35, 'lightgbm': 0.30}
        
        X_scaled = self.scalers['features'].transform(X)
        predictions = {}
        
        # CNN-LSTM prediction
        if 'cnn_lstm' in self.models:
            model = self.models['cnn_lstm']
            model.eval()
            dataset = StockDataset(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            preds = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    X_batch = X_batch.to(next(model.parameters()).device)
                    outputs = model(X_batch)
                    preds.extend(outputs.cpu().numpy())
            predictions['cnn_lstm'] = np.array(preds)
        
        # Transformer prediction
        if 'transformer' in self.models:
            model = self.models['transformer']
            model.eval()
            dataset = StockDataset(X_scaled, np.zeros(len(X_scaled)), self.sequence_length)
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            preds = []
            with torch.no_grad():
                for X_batch, _ in loader:
                    X_batch = X_batch.to(next(model.parameters()).device)
                    outputs = model(X_batch)
                    preds.extend(outputs.cpu().numpy())
            predictions['transformer'] = np.array(preds)
        
        # LightGBM prediction
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = self.models['lightgbm'].predict(X_scaled)
        
        # Weighted ensemble
        ensemble_pred = sum(predictions[name] * weights[name] 
                           for name in predictions.keys())
        
        return ensemble_pred
    
    def save(self, filepath: str):
        """Save ensemble model"""
        import pickle
        
        save_dict = {
            'cluster_id': self.cluster_id,
            'horizon': self.horizon,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'sequence_length': self.sequence_length
        }
        
        # Save PyTorch models
        for name in ['cnn_lstm', 'transformer']:
            if name in self.models:
                torch.save(self.models[name].state_dict(), f"{filepath}_{name}.pth")
        
        # Save LightGBM
        if 'lightgbm' in self.models:
            self.models['lightgbm'].save_model(f"{filepath}_lightgbm.txt")
        
        # Save metadata
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"✓ Model saved to {filepath}")


if __name__ == "__main__":
    print("Model definitions loaded. Use EnsembleModel class to train.")