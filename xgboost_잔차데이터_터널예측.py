#구글 드라이브에 파일 올리고 불러오기

from google.colab import drive
drive.mount('/content/drive')

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


#예측해야하는 구간 data불러오기

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

# 위도와 경도 예측

for i in range(len(ternaltime_pred)):

    # 'Time' 열을 기준으로 해당 시간과 일치하는 행 추출
    start_index = ternaltime_pred[i][1]  # 첫 번째 인덱스 가져오기
    end_index = ternaltime_pred[i][2]
    back_data = df.loc[start_index - 7: start_index , ['Time', 'Latitude', 'Longitude']]
    selected_data = df.loc[start_index - 7: start_index , ['Time', 'Latitude Difference', 'Longitude Difference']] # 예측값에 넣을 데이터 직전 관측값 10개
    additional_data = df.loc[end_index : end_index + 1, ['Time', 'Latitude Difference', 'Longitude Difference']]
    selected_data = pd.concat([selected_data, additional_data])
    predict_input = selected_data.values

    predict_time = ternaltime_pred[i][0]  # 예측 시간 계산 10초 단위로
    # predict_time 대한 위도와 경도 예측값을 얻는 코드 (model_latitude, model_longitude를 사용)

    pred_latitude = model_latitude.predict(predict_input)
    pred_longitude = model_longitude.predict(predict_input)
    time_predictions.append(predict_time)
    pred_latitude = round((sum(pred_latitude) / 10), 7)
    pred_longitude = round((sum(pred_longitude) / 10),7)
    next_prediction_input = np.array([[predict_time, pred_latitude, pred_longitude]])
    predict_input = np.concatenate((predict_input, next_prediction_input), axis=0)

    # pred_latitude 값을 이전 관측값에 더해주는 코드
    last_latitude = back_data.iloc[-3]['Latitude']
    real_latitude = pred_latitude + last_latitude

    # pred_longitude 값을 이전 관측값에 더해주는 코드
    last_longitude = back_data.iloc[-3]['Longitude']
    real_longitude = pred_longitude + last_longitude

    latitude_predictions.append(real_latitude)
    longitude_predictions.append(real_longitude)

    for j in range(1,5):
        predict_time = ternaltime_pred[i][0] + (j * 10)  # 예측 시간 계산 10초 단위로
        # predict_time 대한 위도와 경도 예측값을 얻는 코드 (model_latitude, model_longitude를 사용)
        pred_latitude = model_latitude.predict(predict_input)
        pred_longitude = model_longitude.predict(predict_input)
        # print(pred_latitude, pred_longitude)
        time_predictions.append(predict_time)
        pred_latitude = round(sum(pred_latitude) / (10 + j), 7)
        pred_longitude = round(sum(pred_longitude) / (10 + j),7)
        # 현재 예측값을 이전 관측값에 더하여 실제 위도와 경도를 계산
        real_latitude += pred_latitude
        real_longitude += pred_longitude
        # 결과를 리스트에 추가
        latitude_predictions.append(real_latitude)
        longitude_predictions.append(real_longitude)
        next_prediction_input = np.array([[predict_time, pred_latitude, pred_longitude]])
        predict_input = np.concatenate((predict_input, next_prediction_input), axis=0)






# 새로운 데이터프레임 생성
predictions_df = pd.DataFrame({'Latitude': latitude_predictions,
                               'Longitude': longitude_predictions,
                               'Time': time_predictions})

# 결과를 CSV 파일로 저장
predictions_df.to_csv('prediction_resultt.csv', index=False)

