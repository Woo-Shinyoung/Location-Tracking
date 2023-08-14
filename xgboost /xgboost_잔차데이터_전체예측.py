
#구글 드라이브에 파일 올리고 불러오기

from google.colab import drive
drive.mount('/content/drive')


#모듈 import
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datetime import datetime
import datetime
from sklearn.metrics import mean_squared_error

import xgboost as xgb
from xgboost import plot_importance
import csv


data = pd.read_csv("/content/drive/MyDrive/인천 로그파일.txt")
data.head()

# '인천 로그파일.txt'에서 데이터를 읽어서 DataFrame으로 변환

data = []
with open('인천 로그파일.txt', 'r') as file:
    for line in file:
        if 'ALT' in line:
            continue
        line = line.strip().split(';')
        date = line[0]
        date = date[12:-9]
        hour, minute, second = map(int, date.split(':'))
        latitude = float(line[13])
        longitude = float(line[14])
        if longitude == -117.297160:
            continue
        speed = float(line[12])
        time = hour * 3600 + minute * 60 + second  # 시간, 분, 초를 초로 변환
        data.append([time, latitude, longitude, speed])
df = pd.DataFrame(data, columns=['Time', 'Latitude', 'Longitude', 'Speed'])
df.to_csv('original_data.csv', index=False)

#터널인가를 예측 부분

ternaltime_pred = []
cnt = 0
ternaltime_pred = []

# 데이터프레임을 순회하며 위도와 경도가 동일한 횟수를 세고, 5번 이상인 경우 첫 번째 시간을 저장
for i in range(1, len(df)):
    if df['Latitude'][i] == df['Latitude'][i - 1] and df['Longitude'][i] == df['Longitude'][i - 1]:
        cnt += 1
    else:
        if cnt >= 5:
           ternaltime_pred.append((df['Time'][i - cnt - 1], i - cnt + 1, i))
           cnt =  0
        else:
            cnt = 0

# 위도와 경도의 앞뒤 차이를 계산하여 새로운 열로 추가

df['Latitude Difference'] = df['Latitude'].diff()
df['Longitude Difference'] = df['Longitude'].diff()

df = df.dropna()

print(df)

#데이터 분리 x : 시간, 위도, 속도, 경도 y : 위도, 경도

X =df[['Time', 'Latitude Difference', 'Longitude Difference']]
y_latitude = df['Latitude Difference']
y_longitude = df['Longitude Difference']

# XGBoost 회귀 모델을 초기화

model_longitude = xgb.XGBRegressor()
model_latitude = xgb.XGBRegressor()

# 경도 모델 학습

model_longitude.fit(X, y_longitude)

# 위도 모델 학습

model_latitude.fit(X, y_latitude)

# 새로운 데이터프레임 생성을 위한 빈 리스트
latitude_predictions = []
longitude_predictions = []
time_predictions = []
latitude_errors = []
longitude_errors = []

# 예측할 시간 간격 (초)
prediction_interval = 10

# 예측할 시간 범위 설정
start_time = df['Time'].min()
end_time = df['Time'].max()

# 예측할 시간 리스트 생성
prediction_times = np.arange(start_time, end_time, prediction_interval)

# 예측 결과를 저장할 데이터프레임 생성
predictions_df = pd.DataFrame({'time': prediction_times})

for time in prediction_times:
     # 'Time' 열을 기준으로 해당 시간과 일치하는 행 추출
    target_row = df[df['Time'] == time]

     # target_row가 비어있는 경우 예외 처리
    if target_row.empty:
        continue

    # print(ternaltime)
    # 'Time' 열을 기준으로 해당 시간과 일치하는 행 추출
    # 시간, 위도, 경도, 속도 추출
    time = target_row['Time'].values[0]
    Latitude_Difference = target_row['Latitude Difference'].values[0]
    Longitude_Difference = target_row['Longitude Difference'].values[0]

    # 리스트 생성
    data_list = [time, Latitude_Difference, Longitude_Difference]
    latitude = target_row['Latitude'].values[0]
    longitude = target_row['Longitude'].values[0]
    current_time = time + prediction_interval
    input_data = np.array([data_list])
    latitude_pred = model_latitude.predict(input_data)[0]
    longitude_pred = model_longitude.predict(input_data)[0]
    latitude_pred += latitude
    longitude_pred += longitude
    latitude_error = latitude - latitude_pred
    longitude_error = longitude - longitude_pred
    latitude_errors.append(latitude_error)
    longitude_errors.append(longitude_error)
    # 결과를 리스트에 추가
    latitude_predictions.append(latitude_pred)
    longitude_predictions.append(longitude_pred)
    time_predictions.append(current_time)

error_df = pd.DataFrame({'Latitude Error': latitude_errors, 'Longitude Error': longitude_errors})
# 엑셀 파일로 저장

error_df.to_excel('error_data_10.xlsx', index=False)




# 새로운 데이터프레임 생성
predictions_df = pd.DataFrame({'Latitude': latitude_predictions,
                               'Longitude': longitude_predictions,
                               'Time': time_predictions})


predictions_df.to_csv('total_prediction_10second.csv', index=False)
