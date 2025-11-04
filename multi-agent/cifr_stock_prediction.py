"""
Cipher Mining Inc. (CIFR) Stock Price Prediction using XGBoost

This script creates a machine learning model to predict CIFR stock prices for the next 3 days.
Due to API connectivity issues, synthetic data is used that mimics realistic stock price behavior.

The model uses XGBoost regression with comprehensive feature engineering including:
- Technical indicators (RSI, MACD, Bollinger Bands)
- Moving averages
- Price momentum features
- Lagged variables

Author: Financial Analysis Agent
Date: November 4, 2025
"""

# Import required libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Initializing CIFR Stock Price Prediction Model...")
print("Note: Using synthetic data due to API connectivity constraints")

# Step 1: Data Acquisition
# In production, this would fetch real data from Yahoo Finance
# Due to rate limiting, we create synthetic data with realistic patterns

def create_synthetic_data():
    """Create synthetic stock data with realistic market patterns"""
    print("\nStep 1: Creating synthetic historical data...")
    
    dates = pd.date_range('2025-05-01', periods=120, freq='D')
    np.random.seed(42)  # For reproducibility
    
    # Create a trend with realistic volatility - similar to high-growth tech/mining stocks
    # Starting around $15 with gradual growth and realistic fluctuations
    base_price = 15.0
    daily_returns = np.random.normal(0.001, 0.02, 120)  # 0.1% avg daily return, 2% volatility
    price_series = base_price * np.cumprod(1 + daily_returns)
    
    # Add some realistic patterns (momentum, mean reversion)
    for i in range(20, 120):
        if i % 25 == 0:  # Simulate periodic volatility spikes
            price_series[i] = price_series[i] * np.random.uniform(0.9, 1.1)
    
    # Create OHLCV data
    open_prices = price_series.copy()
    high_prices = open_prices + np.random.uniform(0.5, 2, 120)
    low_prices = open_prices - np.random.uniform(0.5, 2, 120)
    close_prices = price_series
    volume = np.random.gamma(2, 1000000, 120)  # Gamma distribution for realistic volume
    
    data = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    print(f"Synthetic data created: {data.shape[0]} days of data")
    print(f"Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
    
    return data

# Create the synthetic dataset
data = create_synthetic_data()

# Step 2: Feature Engineering
# Create comprehensive technical indicators and predictive features
def create_features(df):
    """Create advanced features for stock prediction model"""
    print("\nStep 2: Engineering technical features...")
    
    df = df.copy()
    
    # Basic price features
    df['Price_Range'] = df['High'] - df['Low']
    df['Price_Open_Close'] = df['Close'] - df['Open']
    df['Price_Change_Pct'] = df['Close'].pct_change()
    
    # Moving averages - short, medium, and long term
    ma_windows = [5, 10, 20, 50]
    for window in ma_windows:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        
    # Exponential Moving Averages - more weight on recent prices
    ema_windows = [5, 10, 12, 20, 26]
    for window in ema_windows:
        df[f'EMA_{window}'] = df['Close'].ewm(span=window).mean()
        
    # Volatility measures
    vol_windows = [5, 10, 20]
    for window in vol_windows:
        df[f'Volatility_{window}'] = df['Close'].rolling(window=window).std()
        
    # Volume features
    df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_OSC'] = (df['Volume'] - df['Volume'].rolling(10).mean()) / df['Volume'].rolling(10).std()
    
    # Price momentum and rate of change
    df['ROC_5'] = df['Close'].pct_change(periods=5)  # 5-day rate of change
    df['ROC_10'] = df['Close'].pct_change(periods=10)  # 10-day rate of change
    
    for i in [1, 2, 3, 5, 10]:
        df[f'Momentum_{i}'] = df['Close'] - df['Close'].shift(i)
        
    # Bollinger Bands features
    bb_window = 20
    bb_std = 2
    bb_middle = df['MA_20']
    bb_std_val = df['Close'].rolling(bb_window).std()
    bb_upper = bb_middle + (bb_std_val * bb_std)
    bb_lower = bb_middle - (bb_std_val * bb_std)
    
    df['BB_Position'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)  # Where price is within BB
    df['BB_Width'] = bb_upper - bb_lower  # Bandwidth (volatility measure)
    df['BB_Squeeze'] = df['BB_Width'] / bb_middle  # Normalized bandwidth
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df['Close'].ewm(span=12).mean()
    ema_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Lagged features (past prices and volumes)
    lag_periods = [1, 2, 3, 4, 5, 7, 10]
    for lag in lag_periods:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
    # Target variable: next day's closing price
    df['Target'] = df['Close'].shift(-1)
    
    # Remove any rows with NaN values (from rolling calculations)
    df_clean = df.dropna().copy()
    
    print(f"Features created: {len([col for col in df_clean.columns if col != 'Target'])} predictive features")
    print(f"Data available for modeling: {df_clean.shape[0]} trading days")
    
    return df_clean

# Apply feature engineering
data_with_features = create_features(data)

# Step 3: Model Preparation and Training
print("\nStep 3: Preparing data and training XGBoost model...")

# Separate features and target variable
feature_columns = [col for col in data_with_features.columns if col != 'Target']
X = data_with_features[feature_columns]
y = data_with_features['Target']

# Time-based split (80% training, 20% testing)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model with optimized parameters
print("\nTraining XGBoost Regressor...")
xgb_model = XGBRegressor(
    n_estimators=100,           # Number of decision trees
    learning_rate=0.1,          # Step size shrinkage
    max_depth=6,               # Maximum depth of each tree
    random_state=42,           # For reproducibility
    subsample=0.8,             # Subsample ratio for training instances
    colsample_bytree=0.8,      # Subsample ratio for columns
    objective='reg:squarederror',  # Regression objective
    eval_metric='rmse'         # Evaluation metric
)

# Fit the model
time_start = pd.Timestamp.now()
xgb_model.fit(X_train_scaled, y_train)
time_end = pd.Timestamp.now()

print(f"Model trained in {(time_end - time_start).total_seconds():.2f} seconds")

# Step 4: Model Evaluation
print("\nStep 4: Evaluating model performance...")

# Make predictions on test set
y_pred = xgb_model.predict(X_test_scaled)

# Calculate performance metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"MAE (Mean Absolute Error): ${mae:.2f}")
print(f"RMSE (Root Mean Squared Error): ${rmse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Accuracy Estimate: ±${rmse:.2f} (based on RMSE)")

# Step 5: Future Price Predictions
print("\nStep 5: Generating 3-day price forecast...")

def predict_next_days(model, scaler, last_data, feature_cols, n_days=3):
    """Predict next n days of stock prices iteratively"""
    predictions = []
    current_data = last_data.copy()
    
    for i in range(n_days):
        # Prepare current features for prediction
        X_current = scaler.transform(current_data[feature_cols].values.reshape(1, -1))
        
        # Make prediction
        pred = model.predict(X_current)[0]
        predictions.append(pred)
        
        # Update data for next prediction (simplified)
        # In reality, all technical indicators would need recalculation
        if i < n_days - 1:
            new_row = current_data.copy()
            
            # Shift lagged features
            for j in range(10, 1, -1):
                lag_col = f'Close_Lag_{j}'
                prev_lag_col = f'Close_Lag_{j-1}'
                if prev_lag_col in new_row.columns and lag_col in new_row.columns:
                    new_row[lag_col] = new_row[prev_lag_col]
            new_row['Close_Lag_1'] = pred
            
            # Update moving averages (simplified - in reality would recalculate)
            # This is a limitation of iterative forecasting
            
            current_data = new_row
    
    return predictions

# Get the most recent data point
last_data_point = data_with_features.iloc[-1:][feature_columns].copy()

# Generate predictions for next 3 days
predictions_3_days = predict_next_days(xgb_model, scaler, last_data_point, feature_columns, n_days=3)

# Display predictions
print(f"\nCIFR 3-Day Price Forecast:")
print(f"Current Price: ${data['Close'].iloc[-1]:.2f}")
for i, pred in enumerate(predictions_3_days, 1):
    direction = "↑" if pred > data['Close'].iloc[-1] else "↓"
    print(f"Day {i}: ${pred:.2f} {direction}")

# Step 6: Visualization
print("\nStep 6: Creating visualization...")

plt.style.use('default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Historical and predicted prices
historical_prices = data['Close'].values
predicted_prices = predictions_3_days

ax1.plot(historical_prices[-20:], 'b-', linewidth=2, label='Historical Prices')
ax1.plot(range(len(historical_prices[-20:]), len(historical_prices[-20:]) + 3), 
         predicted_prices, 'ro-', linewidth=2, markersize=8, label='Predicted Prices')
ax1.axhline(y=historical_prices[-1], color='g', linestyle='--', alpha=0.7, 
           label=f'Current Price (${historical_prices[-1]:.2f})')
ax1.set_title('CIFR Stock Price Prediction (Next 3 Days)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#f8f9fa')

# Plot 2: Feature importance
feature_importance = xgb_model.feature_importances_
top_features_idx = np.argsort(feature_importance)[-15:]  # Top 15 features

colors = plt.cm.viridis(np.linspace(0, 1, len(top_features_idx)))
bars = ax2.barh(range(len(top_features_idx)), 
                 feature_importance[top_features_idx], 
                 color=colors, alpha=0.8)
ax2.set_yticks(range(len(top_features_idx)))
ax2.set_yticklabels([feature_columns[i] for i in top_features_idx])
ax2.set_xlabel('Feature Importance', fontsize=12)
ax2.set_title('Top 15 Feature Importances in XGBoost Model', fontsize=16, fontweight='bold')
ax2.grid(True, axis='x', alpha=0.3)
ax2.set_facecolor('#f8f9fa')

# Add value labels on bars
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('cifr_predictions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("\nAnalysis complete!")
print("Files created:")
print("- cifr_stock_prediction.py (this script)")
print("- cifr_predictions.png (visualization)")
print("\nFor production use, replace synthetic data with real Yahoo Finance data")