"""Forecasting models: SARIMA, LSTM, GRU."""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sequence-to-point forecasting."""
    
    def __init__(self, data, window):
        X, y = [], []
        for i in range(len(data) - window):
            X.append(data[i:i + window])
            y.append(data[i + window])
        self.X = torch.FloatTensor(X).unsqueeze(-1)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """2-layer LSTM for time series forecasting."""
    
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            1, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class GRUModel(nn.Module):
    """2-layer GRU for time series forecasting."""
    
    def __init__(self, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            1, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.fc = nn.Linear(hidden, 1)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def train_nn_model(model, train_scaled, device, epochs=10):
    """Train neural network model."""
    window = 288
    dataset = TimeSeriesDataset(train_scaled[-144*14:], window)  # Last 2 weeks
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    t0 = time.perf_counter()
    
    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            avg_loss = ep_loss / len(loader)
            print(f"    Epoch {epoch+1}/{epochs} loss={avg_loss:.6f}")
    
    return time.perf_counter() - t0


def infer_nn_model(model, test_scaled, device, window=288):
    """Run inference on neural network model."""
    model.eval()
    history = list(test_scaled[:window])
    preds = []
    
    t1 = time.perf_counter()
    with torch.no_grad():
        for _ in range(len(test_scaled) - window):
            inp = torch.FloatTensor(history[-window:]).unsqueeze(0).unsqueeze(-1).to(device)
            p = model(inp).item()
            preds.append(p)
            history.append(p)
    
    return np.array(preds), time.perf_counter() - t1


def run_sarima(train_series, test_series, label, order=(1,0,1), seasonal_order=(1,0,1,144)):
    """Train and forecast with SARIMA."""
    
    scaler = MinMaxScaler()
    train_vals = scaler.fit_transform(train_series.values.reshape(-1, 1)).flatten()
    subset = train_vals[-144*14:]  # Last 2 weeks
    
    t0 = time.perf_counter()
    model = SARIMAX(
        subset,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fit = model.fit(disp=False, maxiter=50)
    t_train = time.perf_counter() - t0
    
    t1 = time.perf_counter()
    forecast_sc = fit.forecast(steps=len(test_series))
    t_exec = time.perf_counter() - t1
    
    forecast = scaler.inverse_transform(forecast_sc.reshape(-1, 1)).flatten()
    print(f"  SARIMA [{label}] train={t_train:.1f}s exec={t_exec:.4f}s")
    
    return forecast, t_train, t_exec


def run_lstm(train_series, test_series, label, window=288, epochs=10):
    """Train and forecast with LSTM."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = MinMaxScaler()
    all_v = np.concatenate([train_series.values, test_series.values])
    scaled = scaler.fit_transform(all_v.reshape(-1, 1)).flatten()
    n = len(train_series)
    
    model = LSTMModel().to(device)
    t_train = train_nn_model(model, scaled[:n], device, epochs)
    preds_sc, t_exec = infer_nn_model(model, scaled[n - window:], device, window)
    
    forecast = scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    print(f"  LSTM  [{label}] train={t_train:.1f}s exec={t_exec:.4f}s")
    
    return forecast, t_train, t_exec


def run_gru(train_series, test_series, label, window=288, epochs=10):
    """Train and forecast with GRU."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = MinMaxScaler()
    all_v = np.concatenate([train_series.values, test_series.values])
    scaled = scaler.fit_transform(all_v.reshape(-1, 1)).flatten()
    n = len(train_series)
    
    model = GRUModel().to(device)
    t_train = train_nn_model(model, scaled[:n], device, epochs)
    preds_sc, t_exec = infer_nn_model(model, scaled[n - window:], device, window)
    
    forecast = scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
    print(f"  GRU   [{label}] train={t_train:.1f}s exec={t_exec:.4f}s")
    
    return forecast, t_train, t_exec
