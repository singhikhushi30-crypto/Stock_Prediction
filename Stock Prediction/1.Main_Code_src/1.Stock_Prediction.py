import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_data(df, forecast_col, forecast_out, test_size):
    df = df.copy()

    for i in range(1, 6):
        df[f'{forecast_col}_lag{i}'] = df[forecast_col].shift(i)

    feature_cols = [f'{forecast_col}_lag{i}' for i in range(1, 6)]

    df = df.dropna(subset=feature_cols)

    df['target'] = df[forecast_col].shift(-forecast_out)

    df = df.dropna(subset=['target'])

    X = df[feature_cols].values
    y = df['target'].values

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    last_rows = df[feature_cols].tail(forecast_out).values
    X_lately = scaler.transform(last_rows)

    return X_train, X_test, y_train, y_test, X_lately, df


print("Downloading GOOG data...")
df = yf.download('GOOG', start='2010-01-01', end='2024-01-01', progress=False)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if 'Close' not in df.columns:
    raise ValueError("Close column not found in data!")

df = df[['Close']].rename(columns={'Close': 'close'}).dropna()

if df.empty:
    raise ValueError("No data fetched. Check internet or ticker.")

print(f"Loaded {len(df)} rows")


forecast_col = 'close'
forecast_out = 5
test_size = 0.2


X_train, X_test, y_train, y_test, X_lately, df_proc = prepare_data(
    df, forecast_col, forecast_out, test_size
)


model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

forecast = model.predict(X_lately)

print(f"Train R2: {train_score:.3f}")
print(f"Test R2: {test_score:.3f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("Next 5 day prediction:", forecast)

plt.figure(figsize=(12, 6))

last_n = 100
plt.plot(df_proc.index[-last_n:], df_proc['close'].tail(last_n), label='Actual')

future_dates = pd.date_range(start=df_proc.index[-1], periods=forecast_out + 1, freq='B')[1:]
plt.plot(future_dates, forecast, label='Predicted', marker='o')

plt.legend()
plt.title('GOOG Stock Prediction (Linear Regression)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("✅ Prediction complete!")