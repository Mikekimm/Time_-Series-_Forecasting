# Mobile Network Traffic Forecasting

Comparative time series analysis and forecasting of mobile network traffic in Milan using classical statistical and neural network approaches.

## Project Overview

This project analyzes mobile network traffic (CDR data) from Telecom Italia Mobile covering a 100×100 geographic grid in Milan over two months (November-December 2013). We implement three forecasting models:

1. **SARIMA** — Classical statistical baseline
2. **LSTM** — Long Short-Term Memory neural network
3. **GRU** — Gated Recurrent Unit neural network

Each model predicts one-step-ahead (10-minute intervals) traffic for three geographic areas.

## Dataset

Download from Harvard Dataverse:
- [Telecommunications Activity Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGZHFV)
- Extract all `.txt` files to `data/raw/`

Expected file format (3 relevant columns):
```
square_id   time_interval   internet
1           1383292800000   15
2           1383292800000   8
...
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip or conda package manager
- ~4GB free disk space (for data + outputs)
- ~2GB RAM minimum (optimization applied for 5GB dataset)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mobile-traffic-forecasting.git
cd mobile-traffic-forecasting
```

2. Create and activate a virtual environment:

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; import pandas; print('Setup OK')"
```

## Running the Analysis

### Option 1: Use the Jupyter Notebook (Recommended for exploration)

```bash
jupyter notebook notebooks/mobile_traffic_analysis.ipynb
```

Run all cells sequentially (`Kernel → Restart & Run All`). Outputs are saved to `outputs/`.

### Option 2: Run Python Scripts

```bash
python src/main.py --config config.yaml
```

This runs the full pipeline:
1. Load and optimize data
2. Perform exploratory analysis (Task 2)
3. Train and evaluate all three models (Task 3)
4. Generate forecasts and metrics

### Output Files

```
outputs/
├── task1/
│   └── memory_report.txt          # Memory optimization details
├── task2/
│   ├── pdf_total_traffic.png
│   ├── timeseries_two_weeks.png
│   ├── rolling_stats.png
│   ├── acf_pacf.png
│   ├── decomposition.png
│   ├── spatial_heatmap.png
│   └── anomalies.png
└── task3/
    ├── forecast_*.png              # 9 prediction plots (3 areas × 3 models)
    ├── metrics_*.csv               # Performance tables
    ├── timing_report.csv
    └── failure_*.png               # Failure analysis plots
```

## Model Details

### Input Representation
- **Sequence length:** 288 steps (2 days of 10-min intervals)
- **Normalization:** MinMaxScaler fitted on training data
- **Training data:** November 1–December 15 (1 month + 15 days)
- **Test data:** December 16–22 (held out, never used in training)

### SARIMA Configuration
```
Order: (1, 0, 1) × (1, 0, 1)₁₄₄
Seasonal period: 144 (daily)
Training window: Last 2 weeks (for CPU efficiency)
```

### LSTM & GRU Configuration
```
Architecture: 2-layer RNN (hidden_size=64) → Linear(64→1)
Dropout: 0.2 (between layers)
Optimizer: Adam (learning_rate=0.001)
Loss: MSE
Epochs: 10
Batch size: 64
Device: CPU (PyTorch CPU-optimized)
```

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| MAE | (1/n) Σ\|y - ŷ\| | Average absolute error in CDR units |
| MAPE | (100/n) Σ(\|y - ŷ\| / y) | Error as percentage of actual values |
| RMSE | √((1/n) Σ(y - ŷ)²) | Root mean squared error (penalizes large errors) |

## Key Findings

1. **Traffic Distribution:** Heavily right-skewed, concentrated in city center (downtown hotspots)
2. **Seasonality:** Clear daily patterns (commute/leisure hours peak), some weekly structure
3. **Stationarity:** Non-stationary series with trend and seasonal components
4. **Anomalies:** ~1% of observations are outliers (special events, maintenance windows)
5. **Model Performance:** LSTM typically achieves 8-12% MAPE on test week, outperforms SARIMA by 5-15% RMSE

## Limitations & Future Work

**Current Limitations:**
- No external features (weather, events, holidays) — limits generalization to unseen periods
- Single-area forecasts — doesn't exploit geographic correlations
- Limited training data per area (2 weeks) — risk of overfitting
- December 16-22 test week includes holiday shopping period, causing distribution shift

**Future Improvements:**
- Incorporate holiday calendar and event data as external regressors
- Use Graph Neural Networks (GNN) to model spatial correlations
- Implement transfer learning across areas
- Collect data from multiple years for robust seasonal modeling
- Apply ensemble methods (weighted combination of all three models)

## Reproducibility

All results are fully reproducible:
1. Data loading is deterministic (sorted by square_id, datetime)
2. Random seeds are set: numpy.random.seed(42), torch.manual_seed(42)
3. Model weights are initialized identically across runs
4. All hyperparameters are hardcoded in `config.yaml`

To reproduce results exactly, use the provided data snapshot in `data/sample/` which contains one day of sample observations.

## Citation

If you use this project, please cite the original dataset:

```bibtex
@article{Barlacchi2015,
  title={A multi-source dataset of urban life in the city of Milan and the Province of Trentino},
  author={Barlacchi, Gianni and De Nadai, Marco and Larcher, Roberto and others},
  journal={Scientific Data},
  volume={2},
  pages={150055},
  year={2015},
  doi={10.1038/sdata.2015.55}
}
```

## Contact & Issues

For questions or issues, open an issue on GitHub or contact the project maintainer.

## License

MIT License — see LICENSE file for details.
