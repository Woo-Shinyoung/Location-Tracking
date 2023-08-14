#구글 드라이브에 파일 올리고 불러오기

from google.colab import drive
drive.mount('/content/drive')

#데이터 불러오기


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


#예측해야하는 구간 직전까지의 data불러오기
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
print(df)

#데이터 분리 x : 시간, 위도, 속도, 경도 y : 위도, 경도

X =df[['Time', 'Latitude', 'Longitude', 'Speed']]
y_latitude = df['Latitude']
y_longitude = df['Longitude']

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
time_list = []

# 예측할 시간 간격 (초)
prediction_interval = 10

# 예측할 시간 범위 설정
start_time = df['Time'].min()
end_time = df['Time'].max()

# 예측할 시간 리스트 생성
time_predictions = np.arange(start_time, end_time, prediction_interval)

for time in time_predictions:
     # 'Time' 열을 기준으로 해당 시간과 일치하는 행 추출
    target_row = df[df['Time'] == time]

     # target_row가 비어있는 경우 예외 처리
    if target_row.empty:
        continue

    # 시간, 위도, 경도, 속도 추출
    time = target_row['Time'].values[0]
    latitude = target_row['Latitude'].values[0]
    longitude = target_row['Longitude'].values[0]
    speed = target_row['Speed'].values[0]


    current_time = time + prediction_interval # 현재 시간 계산

    # 리스트 생성
    data_list = [time, latitude, longitude, speed]

    # current_time에 대한 위도와 경도 예측값을 얻는 코드 (model_latitude, model_longitude를 사용)
    # data_list를 2차원 배열로 변환
    input_data = np.array([data_list])

    # 예측
    latitude_pred = model_latitude.predict(input_data)[0]
    longitude_pred = model_longitude.predict(input_data)[0]

    latitude_predictions.append(latitude_pred)
    longitude_predictions.append(longitude_pred)
    time_list.append(current_time)

    # 실제값과 예측값의 차이 계산
    latitude_error = latitude - latitude_pred
    longitude_error = longitude - longitude_pred

    latitude_errors.append(latitude_error)
    longitude_errors.append(longitude_error)


error_df = pd.DataFrame({'Latitude Error': latitude_errors, 'Longitude Error': longitude_errors})

# 엑셀 파일로 저장
error_df.to_excel('error_data.xlsx', index=False)
# 새로운 데이터프레임 생성
predictions_df = pd.DataFrame({'Latitude': latitude_predictions,
                               'Longitude': longitude_predictions,
                               'Time': time_list})


predictions_df.to_csv('prediction_result.csv', index=False)