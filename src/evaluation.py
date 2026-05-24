"""Evaluation metrics and reporting."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_forecasts(results):
    """
    Evaluate all models and return DataFrame of metrics.
    
    Args:
        results: Dict with keys {area_label: {'actual': Series, 'model_name': array}}
    
    Returns:
        Dict mapping area labels to DataFrames of metrics
    """
    metrics_by_area = {}
    
    for label, data in results.items():
        actual = data['actual'].values
        rows = []
        
        for model_name in ['SARIMA', 'LSTM', 'GRU']:
            if model_name not in data:
                continue
            
            pred = data[model_name][:len(actual)]
            
            rows.append({
                'Model': model_name,
                'MAE': round(mae(actual, pred), 4),
                'MAPE%': round(mape(actual, pred), 2),
                'RMSE': round(rmse(actual, pred), 4)
            })
        
        df = pd.DataFrame(rows).set_index('Model')
        metrics_by_area[label] = df
    
    return metrics_by_area


def format_timing_report(timing):
    """
    Convert timing dict to DataFrame.
    
    Args:
        timing: Dict mapping area labels to {model_name: {'train_s': float, 'exec_s': float}}
    
    Returns:
        DataFrame with columns: Area, Model, Train (s), Exec (s)
    """
    rows = []
    for label, model_times in timing.items():
        for model_name, times in model_times.items():
            rows.append({
                'Area': label,
                'Model': model_name,
                'Train (s)': round(times['train_s'], 2),
                'Exec (s)': round(times['exec_s'], 4)
            })
    
    return pd.DataFrame(rows)


def find_failure_periods(results):
    """
    Identify worst prediction for each model and area.
    
    Returns:
        Dict mapping (area, model) to (idx, error, timestamp)
    """
    failures = {}
    
    for label, data in results.items():
        actual = data['actual']
        
        for model_name in ['SARIMA', 'LSTM', 'GRU']:
            if model_name not in data:
                continue
            
            preds = data[model_name][:len(actual)]
            errors = np.abs(actual.values - preds)
            worst_idx = np.argmax(errors)
            
            failures[(label, model_name)] = {
                'index': worst_idx,
                'error': errors[worst_idx],
                'timestamp': actual.index[worst_idx],
                'actual': actual.iloc[worst_idx],
                'predicted': preds[worst_idx]
            }
    
    return failures


def print_metrics_summary(metrics_by_area, timing_df):
    """Print human-readable summary of results."""
    
    print("\n" + "="*70)
    print("FORECASTING RESULTS SUMMARY")
    print("="*70 + "\n")
    
    for area_label, metrics_df in metrics_by_area.items():
        print(f"\n{area_label}")
        print("-" * 50)
        print(metrics_df.to_string())
        best_model = metrics_df['RMSE'].idxmin()
        print(f"\nBest model (lowest RMSE): {best_model}")
    
    print("\n" + "="*70)
    print("TRAINING & EXECUTION TIME")
    print("="*70)
    print(timing_df.to_string(index=False))
    print()
