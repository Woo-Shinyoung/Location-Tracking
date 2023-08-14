import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
import pandas as pd
from pmdarima.arima import auto_arima
import matplotlib.pyplot as plt
import pandas as pd

with open('Serial_Recv_4_26.Log', 'r') as file:
    lines = file.readlines()

# Time, Latitude, Longitude를 가진 딕셔너리 리스트 생성
data = []
for line in lines:
    time = line[line.find('[')+1:line.find(']')]
    
    parts = line.split(';')
    
    latitude = float(parts[13])
    longitude = float(parts[14])
    
    data.append({
        'Time': time,
        'Latitude': latitude,
        'Longitude': longitude
    })

data = pd.DataFrame(data)

# Time 컬럼을 datetime 형식으로 변환
data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S:%f')

print(data)

# ACF 그래프
plt.figure(figsize=(12, 6))
plot_acf(data['Latitude'], lags=15)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Autocorrelation Function (Latitude)')
plt.show()

# PACF 그래프
plt.figure(figsize=(12, 6))
plot_pacf(data['Latitude'], lags=7)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function (Latitude)')
plt.show()

# ACF 그래프
plt.figure(figsize=(12, 6))
plot_acf(data['Longitude'], lags=15)
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.title('Autocorrelation Function (Longitude)')
plt.show()

# PACF 그래프
plt.figure(figsize=(12, 6))
plot_pacf(data['Longitude'], lags=7)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function (Longitude)')
plt.show()

data['Time'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S:%f')

start_time = pd.to_datetime('2023-04-26 17:15:22:166', format='%Y-%m-%d %H:%M:%S:%f')
end_time = pd.to_datetime('2023-04-26 18:19:57:286', format='%Y-%m-%d %H:%M:%S:%f')

filtered_data = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]
filtered_data = filtered_data.reset_index(drop=True)

# 데이터를 시간 순서대로 정렬
filtered_data.sort_values('Time', inplace=True)

# 주파수 정보를 포함한 인덱스 생성
filtered_data.set_index('Time', inplace=True)
filtered_data.index = pd.date_range(start=filtered_data.index[0], periods=len(filtered_data), freq='10S')

print(filtered_data)
# Latitude에 대한 ARIMA 모델 적용
latitude_model = auto_arima(filtered_data['Latitude'].dropna(), seasonal=False)
latitude_model_fit = latitude_model.fit(filtered_data['Latitude'].dropna())

# Longitude에 대한 ARIMA 모델 적용
longitude_model = auto_arima(filtered_data['Longitude'].dropna(), seasonal=False)
longitude_model_fit = longitude_model.fit(filtered_data['Longitude'].dropna())

# 예측 수행
forecast_latitude = latitude_model_fit.predict(n_periods=60)
forecast_longitude = longitude_model_fit.predict(n_periods=60)

# 예측 결과를 데이터프레임으로 변환하여 인덱스 생성
last_time = filtered_data.index[-1]
forecast_index = pd.date_range(start=filtered_data.index[-1], periods=60, freq='10S')
forecast_df = pd.DataFrame({'Latitude': forecast_latitude, 'Longitude': forecast_longitude}, index=forecast_index)

# 예측 결과 데이터프레임을 시간, Latitude, Longitude로 분리
forecast_df['Time'] = forecast_df.index
forecast_df = forecast_df[['Time', 'Latitude', 'Longitude']]

# CSV 파일로 저장
forecast_df['Time'] = '예측값) ' + forecast_df['Time'].astype(str)
forecast_df.to_csv('forecast_results.csv', index=False)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['Latitude'][1:], filtered_data['Longitude'][1:], label='Original Data')
plt.scatter(forecast_df['Latitude'], forecast_df['Longitude'], color='red', label='Forecast Data')

plt.plot(filtered_data['Latitude'][1:].values.tolist() + forecast_df['Latitude'].values.tolist(),
         filtered_data['Longitude'][1:].values.tolist() + forecast_df['Longitude'].values.tolist(),
         color='purple', linestyle='dashed', label='Forecast Path')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()
