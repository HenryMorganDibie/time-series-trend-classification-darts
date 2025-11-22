# 📈 Robust Time-Series Trend Classification Engine

**Project Name:** `time-series-trend-classification-darts`  

This project develops a stable and validated system for classifying future trends (**Increase, Decrease, or Flat**) in financial time series data using statistical modeling (**ARIMA**) and the **Darts** library in Python.  

The primary objective was to establish a reliable baseline model, demonstrate system stability, and achieve high accuracy on a low-volatility asset.

---

## ✅ Final Conclusion & Stability Report

The system successfully achieved its goal using an **ARIMA** model configuration.

| Metric                   | Value                        |
|---------------------------|------------------------------|
| Asset Analyzed            | FNCL 6.5 Bond Price          |
| Model Used                | ARIMA(15, 1, 0)             |
| Forecast Horizon          | 90 Days                      |
| Classification Threshold  | 1.0% change                  |
| Final Robust Accuracy     | ~100%                        |

**Key Finding:**  
The near-100% accuracy validates the extreme stability and low volatility of the FNCL 6.5 bond data. The model consistently predicted a 'Flat' trend (Class 0), which matched the true price movement over the validation period. This system is confirmed as a **stable and validated baseline** for testing against highly volatile assets.

---

## 🛠️ Project Structure & Key Files

| Folder/File                                | Description                                                                 |
|--------------------------------------------|-----------------------------------------------------------------------------|
| `src/train_model.py`                        | ➡️ The Core Engine. Contains data loading, preprocessing, trend classification logic, ARIMA training, and rolling origin backtest evaluation. |
| `notebooks/`                               | Contains Jupyter notebooks detailing the analytical workflow and methodology. |
| `notebooks/1_New_Data_Exploration.ipynb`  | Initial data import, cleansing, and time-series transformation.             |
| `notebooks/2_Classification_Test.ipynb`   | Methodology for testing the classification threshold and confirming data stability. |
| `models/arima_trend_predictor_90d_1pct`   | Serialized ARIMA(15, 1, 0) model used for classification.                  |
| `requirements.txt`                         | Lists all necessary Python dependencies (including darts, numpy, pandas).  |
| `results/classification_report_90d_1pct.txt` | Final TXT report detailing total predictions, correct predictions, and true/predicted classes. |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ installed.

### Clone the Repository

```bash
git clone https://github.com/HenryMorganDibie/time-series-trend-classification-darts.git
cd time-series-trend-classification-darts
```

## Setup Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate # On macOS/Linux
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Full Classification Test

To run the complete training, evaluation, and reporting process:

```bash
python src/train_model.py
```

This will:

- Load and prepare the data from data/FNCL.6.5.xlsx (Note: Data must be placed in this folder).

- Train the ARIMA model.

- Perform the robust rolling classification backtest (using STEP_SIZE = 90).

- Print the final accuracy report to the console.

- Save the detailed report to results/classification_report_90d_1pct.txt.

## ⚙️ Model & Evaluation Configuration

The core parameters are set in `src/train_model.py`:

| Parameter                  | Value        | Purpose                                                                 |
|----------------------------|-------------|-------------------------------------------------------------------------|
| `FORECAST_HORIZON`         | 90          | Number of days to predict into the future.                              |
| `PERCENT_CHANGE_THRESHOLD` | 0.01 (1%)   | Minimum percentage change to classify a trend as Increase (1) or Decrease (2). |
| ARIMA Orders `(p, d, q)`   | (15, 1, 0)  | Chosen configuration for a stable ARIMA model.                          |
| `STEP_SIZE`                | 90          | Determines how often the rolling forecast is run (every 90 days).       |

---

## 🤝 Next Steps & Future Work

The validated stability of this system provides a strong platform for the next phase: testing the model on **volatile assets** to see if the classification accuracy holds up when the trend class is expected to change frequently.
