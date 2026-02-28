import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Create images1 folder if it doesn't exist
if not os.path.exists('images1'):
    os.makedirs('images1')

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Reading data files...")

df = pd.read_csv('household_power_consumption.txt', sep=';', 
                parse_dates={'datetime': ['Date', 'Time']},
                dayfirst=True,
                na_values=['?', 'NA', '']) 

print(f"Data reading completed, shape: {df.shape}")
print("Data preview:")
print(df.head())

print("Setting datetime as index")
df = df.dropna(subset=['datetime'])
df = df.set_index('datetime')

print("Index after setting:")
print(df.index[:3])
print(df.head(3))

print("\nData column names:")
print(df.columns.tolist())

numeric_cols = df.columns.tolist()
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

print("\nData types after conversion:")
print(df[numeric_cols].dtypes)

missing_before = df[numeric_cols].isna().sum()
print("Missing values count (before processing):")
print(missing_before)

df[numeric_cols] = df[numeric_cols].interpolate(method='time')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

missing_after = df[numeric_cols].isna().sum()
print("\nMissing values count (after processing):")
print(missing_after)

if missing_after.sum() == 0:
    print("\nAll missing values have been successfully processed!")


hourly_kwh = df['Global_active_power'].resample('h').sum() * (1/60)

plt.figure(figsize=(12, 4))
plt.plot(hourly_kwh.index, hourly_kwh.values, color='blue')
plt.title('Hourly Power Consumption (kWh)')
plt.ylabel('kWh')
plt.xlabel('Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('images1/hourly_power_consumption.png')
plt.close()


df_hourly = pd.DataFrame({'Global_active_power': hourly_kwh})
df_hourly['delta_power'] = df_hourly['Global_active_power'].diff().fillna(0)

fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df_hourly.index, df_hourly['Global_active_power'].values, color='blue', label='Consumption (kWh)')
ax2 = ax1.twinx()
ax2.plot(df_hourly.index, df_hourly['delta_power'].values, color='orange', alpha=0.7, label='Delta')
plt.title('Power Consumption and Delta Feature')
plt.legend()
plt.savefig('images1/power_consumption_delta.png')
plt.close()

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df_hourly)

joblib.dump(scaler, 'scaler_electricity.pkl')
print("Scaler has been saved")


look_back = 168        
forecast_horizon = 168 

def create_sequences_multivariate(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[(i + look_back):(i + look_back + forecast_horizon), 0])
    return np.array(X), np.array(y)

X, y = create_sequences_multivariate(scaled_features, look_back, forecast_horizon)

print(f"\nSequence shapes:")
print(f"- X: {X.shape} (number of samples, time steps, number of features)")
print(f"- y: {y.shape} (number of samples, prediction length)")


plt.figure(figsize=(12, 6))
plt.plot(X[0][:, 0], label='Global Active Power (Normalized)')
plt.plot(X[0][:, 1], label='Power Change (Normalized)')
plt.title('Example of Input Sequence (168-hour lookback)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.legend()
plt.grid(True)
plt.savefig('images1/input_sequence_example.png')
plt.close()

num_features = X.shape[2]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)


class ElectricityForecastModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, lstm_layers=1, output_size=168): 
        super(ElectricityForecastModel, self).__init__()
        self.conv1d = nn.Conv1d(input_size, 64, kernel_size=5) 
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.2)
        
        seq_len_after_conv = (look_back - 5 + 1) // 2
        
        self.bi_lstm = nn.LSTM(64, hidden_size, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        
        self.lstm = nn.LSTM(hidden_size*2, 32, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        
        x = self.conv1d(x)
        x = self.max_pool(x)
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 1)
        
        x, _ = self.bi_lstm(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout3(x)
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


model = ElectricityForecastModel(input_size=num_features).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 50
batch_size = 32
best_val_loss = float('inf')
patience = 10
early_stop_counter = 0

from torch.utils.data import TensorDataset, DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test).item()
        val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'electricity_forecast_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

model.load_state_dict(torch.load('electricity_forecast_model.pth'))

plt.figure()
plt.plot(train_losses, label='Train')
plt.plot(val_losses, label='Validation')
plt.title('Model Training & Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('images1/training_validation_loss.png')
plt.close()

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test).cpu().numpy()

y_test_np = y_test.cpu().numpy()

y_pred_unscaled = np.zeros((y_pred_scaled.shape[0] * y_pred_scaled.shape[1], 2))
y_pred_unscaled[:, 0] = y_pred_scaled.reshape(-1)
y_pred_unscaled = scaler.inverse_transform(y_pred_unscaled)[:, 0]

y_test_unscaled = np.zeros((y_test_np.shape[0] * y_test_np.shape[1], 2))
y_test_unscaled[:, 0] = y_test_np.reshape(-1)
y_test_unscaled = scaler.inverse_transform(y_test_unscaled)[:, 0]

global_mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
global_rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))
print(f"Global MAE: {global_mae:.3f} kWh")
print(f"Global RMSE: {global_rmse:.3f} kWh")

model.eval()
with torch.no_grad():
    last_input = torch.FloatTensor(scaled_features[-look_back:].reshape(1, look_back, num_features)).to(device)
    forecast_scaled = model(last_input).cpu().numpy()


forecast = np.zeros((forecast_horizon, 2))
forecast[:, 0] = forecast_scaled.reshape(-1)
forecast = scaler.inverse_transform(forecast)[:, 0]


forecast_index = pd.date_range(start=hourly_kwh.index[-1] + pd.Timedelta(hours=1),
                             periods=forecast_horizon, freq='H')
forecast_df = pd.DataFrame(forecast, index=forecast_index,
                         columns=['Predicted Global Active Power'])

prediksi_kwh_per_hari = forecast_df['Predicted Global Active Power'].resample('D').sum()
print("\nPredicted daily power consumption (kWh):")
print(prediksi_kwh_per_hari)

plt.figure(figsize=(10, 4))
plt.plot(prediksi_kwh_per_hari.index, prediksi_kwh_per_hari.values,
         marker='o', linestyle='-', color='darkgreen')
plt.title('Prediction: Daily Power Consumption (Next 7 Days)')
plt.xlabel('Date')
plt.ylabel('Total (kWh)')
plt.grid(True)
plt.tight_layout()
plt.savefig('images1/predicted_daily_consumption.png')
plt.close()

historical = hourly_kwh[-look_back:]
plt.figure(figsize=(15, 5))
plt.plot(historical.index, historical.values, label='Last 7 Days')
plt.plot(forecast_df.index, forecast_df['Predicted Global Active Power'],
         label='Next 7 Days Prediction', linestyle='--')
plt.title('Power Consumption Forecast')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.grid(True)
plt.savefig('images1/power_consumption_forecast.png')
plt.close()

with torch.no_grad():
    test_input = torch.FloatTensor(scaled_features[-(look_back + forecast_horizon):-forecast_horizon].reshape(1, look_back, num_features)).to(device)
    pred_scaled = model(test_input).cpu().numpy()

actual_output = scaled_features[-forecast_horizon:, 0]

pred = np.zeros((forecast_horizon, 2))
pred[:, 0] = pred_scaled.reshape(-1)
pred = scaler.inverse_transform(pred)[:, 0]

actual = np.zeros((forecast_horizon, 2))
actual[:, 0] = actual_output
actual = scaler.inverse_transform(actual)[:, 0]

pred_index = hourly_kwh.index[-forecast_horizon:]
comparison_df = pd.DataFrame({'Actual': actual, 'Predicted': pred}, index=pred_index)


plt.figure(figsize=(15, 5))
plt.plot(comparison_df.index, comparison_df['Actual'], label='Actual')
plt.plot(comparison_df.index, comparison_df['Predicted'], label='Predicted', linestyle='--')
plt.title('Backtest: Predicted vs Actual (Hourly, 7 Days)')
plt.xlabel('Date and Time')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.grid(True)
plt.savefig('images1/backtest_hourly_comparison.png')
plt.close()


comparison_daily = comparison_df.resample('D').sum()
plt.figure(figsize=(10, 4))
plt.plot(comparison_daily.index, comparison_daily['Actual'], label='Actual Daily', marker='o')
plt.plot(comparison_daily.index, comparison_daily['Predicted'], label='Predicted Daily', marker='x', linestyle='--')
plt.title('Backtest: Predicted vs Actual Daily Consumption')
plt.xlabel('Date')
plt.ylabel('Total (kWh)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('images1/backtest_daily_comparison.png')
plt.close()

mae = mean_absolute_error(comparison_daily['Actual'], comparison_daily['Predicted'])
rmse = np.sqrt(mean_squared_error(comparison_daily['Actual'], comparison_daily['Predicted']))
print("\nDaily total comparison (kWh):")
print(comparison_daily)
print(f"\nDaily MAE: {mae:.3f} kWh")
print(f"Daily RMSE: {rmse:.3f} kWh")
