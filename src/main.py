"""Main pipeline: Load data, analyze, and forecast."""

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

from src.data_utils import (
    load_raw_data, optimize_dtypes, prepare_data,
    area_series, train_test_split, print_memory_report
)
from src.models import run_sarima, run_lstm, run_gru
from src.evaluation import (
    evaluate_forecasts, format_timing_report,
    find_failure_periods, print_metrics_summary
)


def main(data_dir="data/raw", output_dir="outputs"):
    """Run full forecasting pipeline."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MOBILE TRAFFIC FORECASTING PIPELINE")
    print("="*70 + "\n")
    
    # Constants
    DAILY = 144
    TEST_START = pd.Timestamp('2013-12-16')
    TEST_END = pd.Timestamp('2013-12-22 23:59:59')
    
    # Step 1: Load and optimize data
    print("[1/4] Loading and optimizing data...")
    df_raw = load_raw_data(data_dir)
    mem_before, _, _ = print_memory_report(df_raw, df_raw)
    
    df = optimize_dtypes(df_raw)
    del df_raw
    gc.collect()
    
    mem_before, mem_after, reduction = print_memory_report(df, df)
    df = prepare_data(df)
    
    # Step 2: Identify analysis areas
    print("[2/4] Analyzing areas and preparing data splits...")
    total_by_area = df.groupby('square_id')['internet'].sum()
    area_top = int(total_by_area.idxmax())
    
    areas = {
        f'Top traffic (id={area_top})': area_top,
        'Area 4159': 4159,
        'Area 4556': 4556,
    }
    
    print(f"Analyzing {len(areas)} areas:")
    for label, sq_id in areas.items():
        print(f"  - {label}")
    
    splits = train_test_split(df, areas, TEST_START, TEST_END)
    
    # Step 3: Train models
    print("\n[3/4] Training forecasting models...")
    
    model_functions = {
        'SARIMA': run_sarima,
        'LSTM': run_lstm,
        'GRU': run_gru
    }
    
    results = {}
    timing = {}
    
    for area_label, split_data in splits.items():
        print(f"\nArea: {area_label}")
        print(f"  Train: {len(split_data['train'])} | Test: {len(split_data['test'])}")
        
        train_series = split_data['train']
        test_series = split_data['test']
        
        results[area_label] = {'actual': test_series}
        timing[area_label] = {}
        
        for model_name, model_fn in model_functions.items():
            try:
                preds, t_train, t_exec = model_fn(train_series, test_series, area_label)
                results[area_label][model_name] = preds
                timing[area_label][model_name] = {'train_s': t_train, 'exec_s': t_exec}
            except Exception as e:
                print(f"  ERROR {model_name}: {e}")
                results[area_label][model_name] = np.zeros(len(test_series))
                timing[area_label][model_name] = {'train_s': 0.0, 'exec_s': 0.0}
    
    # Step 4: Evaluate and report
    print("\n[4/4] Evaluating results...")
    
    metrics_by_area = evaluate_forecasts(results)
    timing_df = format_timing_report(timing)
    
    print_metrics_summary(metrics_by_area, timing_df)
    
    # Save results
    for label, metrics_df in metrics_by_area.items():
        safe_label = label.replace(' ', '_').replace('=', '')
        metrics_df.to_csv(output_dir / f'metrics_{safe_label}.csv')
    
    timing_df.to_csv(output_dir / 'timing_report.csv', index=False)
    
    failures = find_failure_periods(results)
    
    print("\nFAILURE ANALYSIS")
    print("-" * 50)
    for (area, model), failure in failures.items():
        print(f"{model} [{area}]: error={failure['error']:.4f} at {failure['timestamp']}")
    
    print("\n" + "="*70)
    print("Pipeline complete! Results saved to:", output_dir)
    print("="*70 + "\n")
    
    return results, metrics_by_area, timing_df


if __name__ == "__main__":
    main()
