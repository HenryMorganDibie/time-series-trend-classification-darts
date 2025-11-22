import pandas as pd
import numpy as np
import os
import pickle
from datetime import timedelta
from darts import TimeSeries
from darts.models import ARIMA
from darts.utils.model_selection import train_test_split 
from collections import Counter

# --- CONFIGURATION ---
DATA_PATH = "data/FNCL.6.5.xlsx"
MODEL_PATH = "models/arima_trend_predictor_90d_1pct" 
RESULTS_PATH = "results/classification_report_90d_1pct.txt"

FORECAST_HORIZON = 90
TRAIN_SPLIT_PERCENT = 0.80
STEP_SIZE = 1 
PERCENT_CHANGE_THRESHOLD = 0.01 

# ------------------------------------------------------------------
# --- TREND CLASSIFICATION ---
# ------------------------------------------------------------------
def classify_trend(series: TimeSeries, threshold: float) -> int:
    """Classify trend based on percentage change between first vs last 5-day average. 
        0=Flat, 1=Increase, 2=Decrease."""
    window = 5
    if len(series) < window * 2:
        return 0
    
    V_start = series[:window].values().mean()
    V_end = series[-window:].values().mean()
    change = V_end - V_start
    
    if V_start == 0:
        return 0 
        
    percent_change = change / V_start
    
    if percent_change > threshold:
        return 1
    elif percent_change < -threshold:
        return 2
    else:
        return 0

# --- ANALYSIS FUNCTION (FOR DIAGNOSTICS) ---
def analyze_class_distribution(series_list, title, threshold):
    """Analyzes and prints the distribution of trend classes in a list of TimeSeries."""
    classes = [classify_trend(s, threshold) for s in series_list]
    counts = Counter(classes)
    
    total = len(classes)
    
    print(f"\n--- Class Distribution for {title} (Threshold={threshold*100:.4f}%) ---")
    print(f"Total Samples: {total}")
    print(f"Class 0 (Flat): {counts[0]} ({counts[0]/total*100:.2f}%)")
    print(f"Class 1 (Increase): {counts[1]} ({counts[1]/total*100:.2f}%)")
    print(f"Class 2 (Decrease): {counts[2]} ({counts[2]/total*100:.2f}%)")
    print("------------------------------------------")
    return classes

# ------------------------------------------------------------------
# --- DATA LOADING & PREPARATION (Unchanged Logic) ---
# ------------------------------------------------------------------
def load_and_prepare_data(path):
    print("Loading and preparing data...")
    
    df = pd.read_excel(path)

    df.columns = df.columns.str.strip()
    date_col = 'Dates' 
    value_col = 'FNCL 6.5 2022 FL Mtge'
    
    if date_col in df.columns and value_col in df.columns:
        df = df[[date_col, value_col]].copy()
    else:
        print("Warning: Expected column names not found after reading Excel. Assuming Date is first column, Value is second.")
        if len(df.columns) >= 2:
             df = df.iloc[:, :2].copy()
             df.columns = [date_col, value_col]
        else:
             raise ValueError("File is too malformed; cannot isolate the two required columns.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df.set_index(date_col, inplace=True)
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    df.dropna(subset=[value_col], inplace=True)
    df = df[[value_col]].sort_index()

    if not df.index.is_unique:
        df = df.groupby(df.index).mean()
        
    df[value_col] = df[value_col].interpolate(method='linear')
    
    series = TimeSeries.from_dataframe(df, value_cols=value_col, freq='D')
    train_series, val_series = train_test_split(series, test_size=1.0 - TRAIN_SPLIT_PERCENT)

    print(f"Using trend classification threshold: {PERCENT_CHANGE_THRESHOLD*100:.4f}%")
    return series, train_series, val_series, PERCENT_CHANGE_THRESHOLD

# ------------------------------------------------------------------
# --- MODEL TRAINING (ARIMA Fix) ---
# ------------------------------------------------------------------
def train_model(train_series):
    print("\nTraining ARIMA Model...")
    
    model = ARIMA(
        p=15, 
        d=1,  
        q=0   
    )
    
    model.fit(train_series)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model

# ------------------------------------------------------------------
# --- EVALUATION (ARIMA Prediction) ---
# ------------------------------------------------------------------
def evaluate_classification(model, full_series, train_split_index, threshold):
    print("\nEvaluating classification accuracy with rolling origin backtest...")
    
    first_val_date = full_series.time_index[train_split_index]
    
    backtest_dates = [
        date for date in full_series.time_index 
        if date >= first_val_date and date <= full_series.time_index[-1] - timedelta(days=FORECAST_HORIZON)
    ]
    
    backtest_dates = backtest_dates[::STEP_SIZE]

    # --- Class Distribution Analysis ---
    validation_windows = []
    for date in backtest_dates:
        true_series = full_series.slice(
            date + timedelta(days=1),
            date + timedelta(days=FORECAST_HORIZON)
        )
        if len(true_series) == FORECAST_HORIZON:
             validation_windows.append(true_series)

    analyze_class_distribution(validation_windows, "Validation Set True Labels", threshold)

    true_classes = []
    predicted_classes = []

    print(f"Total rolling prediction points: {len(backtest_dates)}")

    for i, date in enumerate(backtest_dates):
        history = full_series[:date]
        
        forecast = model.predict(n=FORECAST_HORIZON, series=history, verbose=False)

        true_series = full_series.slice(
            date + timedelta(days=1),
            date + timedelta(days=FORECAST_HORIZON)
        )
        
        if len(true_series) < FORECAST_HORIZON:
            continue

        predicted_classes.append(classify_trend(forecast, threshold))
        true_classes.append(classify_trend(true_series, threshold))

    correct = sum(1 for p, t in zip(predicted_classes, true_classes) if p == t)
    total = len(true_classes)
    accuracy = correct / total if total > 0 else 0

    print("\n--- Classification Results ---")
    print(f"Total Predictions: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Robust Accuracy: {accuracy*100:.2f}%")

    os.makedirs("results", exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Robust Classification Accuracy (STEP_SIZE={STEP_SIZE}): {accuracy*100:.2f}%\n")
        f.write(f"Total Predictions: {total}\n")
        f.write(f"Correct Predictions: {correct}\n")
        f.write(f"True Classes: {true_classes}\n")
        f.write(f"Predicted Classes: {predicted_classes}\n")
    print(f"Report saved to {RESULTS_PATH}")
    return accuracy

# ------------------------------------------------------------------
# --- MAIN ---
# ------------------------------------------------------------------
if __name__ == "__main__":
    try:
        os.makedirs("data", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        full_series, train_series, val_series, threshold = load_and_prepare_data(DATA_PATH)
        
        if len(train_series) < FORECAST_HORIZON:
             print("Error: Training series is too short for the configured forecast horizon. Please check your dataset.")
        else:
             model = train_model(train_series)
             train_split_index = len(full_series) - len(val_series)
             accuracy = evaluate_classification(model, full_series, train_split_index, threshold)

             if accuracy >= 0.70:
                 print("\n GOAL ACHIEVED: Accuracy >= 70%")
             else:
                 print("\n Accuracy below 70%. Consider tuning parameters (e.g., ARIMA orders p, d, q, or PERCENT_CHANGE_THRESHOLD).")

    except Exception as e:
        print(f"Unexpected error: {e}")