# Mobile Network Traffic Forecasting

Comparative time series analysis and forecasting of mobile network traffic in Milan using classical statistical and neural network approaches.

## Project Overview

This project looks at real mobile internet traffic data from Telecom Italia Mobile across a 100×100 grid covering the city of Milan over two months, November to December 2013. I built and compared three models to forecast traffic one step ahead at 10-minute intervals across three different geographic areas:

SARIMA — statistical baseline
LSTM — Long Short-Term Memory network
GRU — Gated Recurrent Unit network

Each model predicts one-step-ahead (10-minute intervals) traffic for three geographic areas.

## Dataset

Download from Harvard Dataverse:
- Telecommunications Activity Dataset
- Extract all `.txt` files to `data/raw/`



## Setup Instructions

# 1. Clone and enter the repo
git clone https://github.com/yourusername/mobile-traffic-forecasting.git
cd mobile-traffic-forecasting

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import torch; import pandas; print('Setup OK')"
### Prerequisites
- Python 3.8+
- pip or conda package manager
- ~4GB free disk space (for data + outputs)
- ~2GB RAM minimum (optimization applied for 5GB dataset)

### Installation


## Running the Analysis
Notebook

jupyter notebook notebooks/mobile_traffic_analysis.ipyn

Output Structure
outputs/
├── task1/  memory_report.txt
├── task2/  pdf_total_traffic.png, timeseries_two_weeks.png,
│           rolling_stats.png, acf_pacf.png, decomposition.png,
│           spatial_heatmap.png, anomalies.png
└── task3/  forecast_*.png (9 plots), metrics_*.csv,
            timing_report.csv, failure_*.png


Disadvanatages

No external features (holidays, weather, events) which was the  main cause of December forecast errors
Single-area models don't exploit geographic correlations between grid cells
Future work: calendar regressors, Graph Neural Networks for spatial modelling, multi-year training data

Citations 

@article{Barlacchi2015,
  title={A multi-source dataset of urban life in the city of Milan and the Province of Trentino},
  author={Barlacchi, Gianni and De Nadai, Marco and Larcher, Roberto and others},
  journal={Scientific Data}, volume={2}, pages={150055}, year={2015},
  doi={10.1038/sdata.2015.55}
}

**Video Demo Link:** https://youtu.be/Gj4_5xqe4zM

**Dataset: ** https://github.com/Mikekimm/Time_-Series-_Forecasting.git
