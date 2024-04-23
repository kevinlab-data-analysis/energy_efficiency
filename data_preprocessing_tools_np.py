import pymysql
import csv
from datetime import datetime, timedelta
import os
import re
import numpy as np
import yaml
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.covariance import MinCovDet
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances 
import joblib
import copy
from collections import defaultdict
from scipy.optimize import minimize
import csv
import matplotlib.pyplot as plt
from prophet import Prophet


'''
application all data form
: Read_Data_in_DB, Extract_Usage_Datta, Except_None_Data

only finedust(in weather)

'''

class lbems_data_tools:
    ###### application all data set DB version tools
    def __init__(self, config):
        self.config = config


    def Read_Data_in_DB(self, host, db, query):
        conn = pymysql.connect(
            host=host,
            user='viewonly',
            passwd='db_view!@09',
            db=db
        )

        curs = conn.cursor()
        curs.execute(query)
        total_result = curs.fetchall()
        conn.close()
        total_result = np.array(total_result)
        
        return total_result

    def Read_Data_in_DB_new(self, host, db, query, columns, convert_type):
        conn = pymysql.connect(
            host=host,
            user='viewonly',
            passwd='db_view!@09',
            db=db
        )

        df = pd.read_sql_query(query, conn)
        conn.close()
        
        for col in columns:
            column_name = df.columns[col]
            df[column_name] = df[column_name].astype(convert_type)

        # 결과를 numpy 배열로 변환
        total_result_np = df.to_numpy()
        
        # print(total_result_np)
        return total_result_np
    
    def Extract_Usage_Data(self, total_result, cnt_list):
        # NumPy 배열의 슬라이싱을 이용하여 데이터 추출
        extracted_data = total_result[:, cnt_list]
        
        return extracted_data

    def Except_None_Data(self, extracted_data):
        # NumPy의 불리언 인덱싱을 이용하여 None이나 빈 문자열을 제외
        mask = ~np.isin(extracted_data, [None, ''])
        except_extracted_data = extracted_data[np.all(mask, axis=1)]
        
        return except_extracted_data

###### meter_finedust(in weather) 적용 tools
    ## date : [[2020121300, 2020121301, 2020121302...], [], ...[]]
    ## val : [[tem2020121300, hum2020121300], [tem2020121301, hum2020121301], [tem2020121302, hum2020121302]...]
    ## 시간대별 첫번째값을 가져옴
    def Make_Finedust_Data(self, except_extracted_data):
        finedust_data = {}
        for row in except_extracted_data:
            year = row[0].year
            month = row[0].month
            day = row[0].day
            hour = row[0].hour

            key = (year, month, day, hour)
            date_format = '%Y%m%d%H'
            if key not in finedust_data:
                finedust_data[key] = (row[0].strftime(date_format), row[1], row[2])

        # finedust_data의 값들을 리스트로 변환 후 NumPy 배열로 변환
        finedust_data_list = list(finedust_data.values())
        # finedust_data_np = np.array(finedust_data_list, dtype=object)

        return finedust_data_list
    
###### csv_weather(out_weather) 적용 tools(csv version)
    def Make_CSV_Path_List(self, raw_data_path):
        csv_path_list = []
        csv_list = os.listdir(raw_data_path)
        for cl in csv_list:
            path = os.path.join(raw_data_path, cl)
            csv_path_list.append(path)
        return csv_path_list
    
    def Make_Full_Data_List(self, csv_path_list):
        full_data = []
        for cpl in csv_path_list:
            with open(cpl, 'r', newline='', encoding='utf-8', errors='replace') as f:
                csv_reader = csv.reader(f)
                # 첫 번째 행(헤더)를 제외하고 데이터 읽기
                next(csv_reader, None)  # 첫 번째 행 건너뛰기
                for row in csv_reader:
                    full_data.append(row)
        
        # 리스트를 NumPy 배열로 변환
        # full_data_array = np.array(full_data)
        return full_data
    
    def make_weather_datasets(self, data_array):
        # 결과 데이터셋을 저장할 리스트
        weather_datasets = []

        # 날씨 상태 코드를 문자열로 매핑
        weather_code_to_string = {
            '01': '맑음',
            '02': '구름 조금',
            '03': '구름 많음',
            '04': '흐림',
            '09': '흐리고 비옴',
            '10': '비옴',
            '11': '뇌우',
            '13': '눈',
            '50': '안개'
        }

        # 데이터셋 생성을 위한 루프
        for data in data_array:
            date = data[0]
            temperatures = data[1:25]  # 1부터 24까지가 온도 데이터
            humidities = data[25:49]  # 25부터 끝까지가 습도 데이터
            weather = data[49:]
            # 하루의 데이터를 저장할 리스트
            daily_weather_data = []

            for hour in range(24):
                hour_str = '{:02d}'.format(hour)  # 시간을 문자열로 변환
                weather_string = weather_code_to_string.get(weather[hour][:-1], "알 수 없음")  # 매핑되지 않은 코드는 "알 수 없음" 처리
                weather_data = [
                    f'{date}{hour_str}',  # 날짜와 시간 결합
                    temperatures[hour],  # 해당 시간의 온도
                    humidities[hour] ,    # 해당 시간의 습도
                    # weather[hour]
                    weather_string
                ]
                daily_weather_data.append(weather_data)
            
            # 각 날짜별로 처리된 데이터를 전체 리스트에 추가
            weather_datasets.extend(daily_weather_data)

        return np.array(weather_datasets)  # NumPy 배열로 반환

    # def Make_Meter_Weather_outside(self, except_extracted_data):
    #     # 예외 처리된 데이터의 shape 확인 (n_rows, n_columns)
    #     n_rows, n_columns = except_extracted_data.shape

    #     # 날짜 데이터 추출 (모든 행의 첫 번째 열)
    #     dates = except_extracted_data[:, 0].astype(str)
        
    #     # 시간대별 날씨 데이터 추출 (온도: 1~24열, 습도: 25~48열)
    #     temperatures = except_extracted_data[:, 1:int(n_columns/2)]
    #     humidities = except_extracted_data[:, int(n_columns/2):]

    #     # 날짜와 시간을 결합하여 새로운 날짜-시간 배열 생성
    #     hours = np.arange(1, int(n_columns/2) + 1)
    #     date_hours = np.char.add(dates[:, None], hours.astype(str).fill(2))

    #     # 최종 데이터 배열 생성
    #     # date_hours, temperatures, humidities 배열을 하나로 결합
    #     output_weather = np.empty((n_rows * (int(n_columns/2)), 3), dtype=object)
    #     for i in range(n_rows):
    #         for j in range(int(n_columns/2)):
    #             output_weather[i * (int(n_columns/2)) + j, 0] = date_hours[i, j]
    #             output_weather[i * (int(n_columns/2)) + j, 1] = temperatures[i, j]
    #             output_weather[i * (int(n_columns/2)) + j, 2] = humidities[i, j]

    #     return output_weather
    
    def Extract_Usage_Data_in_CSV(self, full_data):
        extracted_data = []
        for fd in full_data:
            date = re.split(r'[-: ]', fd[1])
            # print(date)
            if date[4] != '00':
                continue
            else:
                save_date = date[0] + date[1] + date[2] + date[3]
                extracted_data.append([save_date, fd[2], fd[7]])
        extracted_data_array = np.array(extracted_data)
        return extracted_data_array
    
    ###구름 등 날씨 데이터 변환
    def transform_weather_data(self, data):
    # 변환된 데이터를 저장할 빈 리스트 초기화
        transformed_data = []

        # 각 행에 대해 반복
        for row in data:
            val_date = row[0]  # 날짜 데이터
            for hour, value in enumerate(row[1:], start=0):  # 첫 번째 요소는 날짜이므로 두 번째 요소부터 반복
                # YYYYMMDDHH 형태로 날짜와 시간을 결합
                date_hour = datetime.strptime(f"{val_date}{hour:02d}", '%Y%m%d%H')
                # print(date_hour)
                # print(value)
                # 변환된 행 추가
                transformed_data.append([date_hour, value[:-1]])
        
        # 변환된 데이터를 numpy 배열로 변환
        transformed_data_np = np.array(transformed_data)
        
        return transformed_data_np


    def Cold_Sensor_Divide(self, excepted_extracted_data):
        # 센서 ID가 변경되는 지점을 찾습니다.
        sensor_ids = excepted_extracted_data[:, 0]  # 센서 ID 열
        changes = np.where(sensor_ids[:-1] != sensor_ids[1:])[0] + 1  # 센서 ID가 변경되는 인덱스
        division_points = np.split(excepted_extracted_data, changes)  # 변경 지점을 기준으로 배열 분할
        
        # 각 분할된 부분의 데이터를 추출하여 리스트에 추가합니다.
        divided_data = [division[:, 1:] for division in division_points]  # 센서 ID 열을 제외한 데이터 추출
        
        return divided_data
    

    def Calculate_Consumption_cold (self, divided_data):

        boundary_list = []
        model_list = []
        scaler_list = []

        all_processed_data = [] 
        # 5분단위 사용량 데이터로 1시간 사용량 계산
        #  for 문 i 하나당 sensor 1개
        for i in range(len(divided_data)):
            # 각 센서별로 위에서 만든 함수를 거쳐, 5분단위 사용량을 만듦

            five_minute_cold, sum_of_usage  = self.minute_five_cal(divided_data[i])
            # 각 센서별로, outlier 처리한 리스트들을 저장하여, outlier_processing_2을 실행시킴
            processed_outlier_list = []
            processed_outlier1, upper = self.outlier_processing_2(five_minute_cold)
            processed_outlier_list.append(processed_outlier1)
            boundary_list.append([upper])

            processed_outlier2, forest_model, forest_scaler = self.Isolation_Forest_Outlier_filter(five_minute_cold)
            processed_outlier_list.append(processed_outlier2)
            model_list.append([forest_model])
            scaler_list.append([forest_scaler])

            processed_outlier3, auto_model, auto_scaler, auto_threshold = self.AutoEncoder_Outlier_filter(five_minute_cold)
            processed_outlier_list.append(processed_outlier3)
            model_list[i].append(auto_model)
            scaler_list[i].append(auto_scaler)
            boundary_list[i].append(auto_threshold)

            # processed_outlier4, mcd_model, mcd_scaler, mcd_threshold = self.MCD_Outlier_filter(five_minute_cold)
            # processed_outlier_list.append(processed_outlier4)
            # model_list[i].append(mcd_model)
            # scaler_list[i].append(mcd_scaler)
            # boundary_list[i].append(mcd_threshold)

            processed_outlier5, svm_model, svm_scaler = self.OneClassSVM_Outlier_filter(five_minute_cold)
            processed_outlier_list.append(processed_outlier5)
            model_list[i].append(svm_model)
            scaler_list[i].append(svm_scaler)

            processed_outlier_cold = self.outlier_processing(processed_outlier_list)

            # 각 센서별로 outlier가 제외된 데이터를 1시간 단위 사용량으로 변경함
            hour_Consumption = self.Calculate_Consumption_minute_new(processed_outlier_cold)
            filled_hour_Consumption = self.fill_in_hour(hour_Consumption)
            filled_interpolated_hour_Consumption = self.Interpolate_prophet(filled_hour_Consumption, sum_of_usage)
            
            all_processed_data.append(filled_interpolated_hour_Consumption)
        final_df = pd.concat(all_processed_data).groupby('hour_key')['total_usage_hourly'].sum().reset_index()
        return final_df

    def minute_five_cal(self, except_extracted_data):
        # 데이터 프레임 생성
        df = pd.DataFrame(except_extracted_data, columns=['datetime', 'usage'])
        
        # datetime 컬럼을 datetime 타입으로 변환
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d%H%M%S')
        
        # 결과를 저장할 빈 데이터프레임 생성
        calculated_data = []
        sum_of_usage = []

        for i in range(len(df) - 1):
            start_time = df.iloc[i]['datetime']
            end_time = df.iloc[i + 1]['datetime']

            #end time 반올림
            remainder_end = end_time.minute % 5
            if remainder_end in [0, 1, 2]:  # 내림
                adjusted_minutes_end = -(remainder_end)
            else:  # 올림
                adjusted_minutes_end = 5 - remainder_end
            
            adjusted_end_time = end_time + timedelta(minutes=adjusted_minutes_end)
            # 분이 60이 되는 경우 시간 조정
            if adjusted_end_time.minute >= 60:
                adjusted_end_time += timedelta(hours=1)
                adjusted_end_time = adjusted_end_time.replace(minute=0)

            #start time 반올림
            remainder_start = start_time.minute % 5
            if remainder_start in [0, 1, 2]:  # 내림
                adjusted_minutes_start = -(remainder_start)
            else:  # 올림
                adjusted_minutes_start = 5 - remainder_start
            
            adjusted_start_time = start_time + timedelta(minutes=adjusted_minutes_start)
            # 분이 60이 되는 경우 시간 조정
            if adjusted_start_time.minute >= 60:
                adjusted_start_time += timedelta(hours=1)
                adjusted_start_time = adjusted_start_time.replace(minute=0)

            time_difference = (adjusted_end_time - adjusted_start_time).total_seconds()

            usage_diff = df.iloc[i + 1]['usage'] - df.iloc[i]['usage']

            if time_difference <= 300:
                
                new_date = adjusted_end_time.strftime('%Y%m%d%H%M%S')
                calculated_data.append({'original_datetime': df.iloc[i + 1]['datetime'], 'new_datetime': adjusted_end_time, 'new_date': new_date, 'usage_five': usage_diff})

            elif 300 < time_difference < 7200:
                # 시간 차이에 따라 사용량을 나누어 여러 행에 분배
                num_intervals = int(time_difference // 300)  # 5분 간격으로 나누기
                usage_per_interval = usage_diff / num_intervals
                for j in range(num_intervals):
                    interval_time = adjusted_start_time + timedelta(minutes=5*(j+1))
                    # 시간을 5분 단위로 반올림
                    interval_minute = interval_time.minute
                    remainder = interval_minute % 5
                    if remainder in [0, 1, 2]:  # 내림
                        adjusted_minutes = -(remainder)
                    else:  # 올림
                        adjusted_minutes = 5 - remainder
                    interval_time += timedelta(minutes=adjusted_minutes)
                    if interval_time.minute >= 60:
                        interval_time += timedelta(hours=1)
                        interval_time = interval_time.replace(minute=0)

                    new_date = interval_time.strftime('%Y%m%d%H%M%S')
                    calculated_data.append({'original_datetime': start_time + timedelta(minutes=5*(j+1)), 'new_datetime': interval_time, 'new_date': new_date, 'usage_five': usage_per_interval})

            elif time_difference >= 7200:
                total_usage = usage_diff  # 총 사용량

                # 시작 시간의 다음 시간대를 계산 (예: 13:05의 다음 시간대는 14:00)
                next_hour = adjusted_start_time + timedelta(hours=1)
                next_hour = next_hour.replace(minute=0, second=0, microsecond=0)

                # sum_of_usage에 총 사용량 기록
                sum_of_usage.append([next_hour.strftime("%Y%m%d%H"), total_usage])
                # 시작 시간부터 끝 시간까지의 모든 5분 간격에 대해 빈 데이터 삽입 (사용량은 기록하지 않음)
                current_time = adjusted_start_time
                while current_time < adjusted_end_time:
                    # 현재 시간을 5분 단위로 조정
                    current_time += timedelta(minutes=5)  
                    if current_time >= adjusted_end_time:  # 끝 시간에 도달하면 반복 중단
                        break

                    # 5분 간격 데이터 추가 (사용량은 0으로 설정)
                    new_date = current_time.strftime('%Y%m%d%H%M%S')
                    calculated_data.append({'original_datetime': current_time, 'new_datetime': current_time, 'new_date': new_date, 'usage_five': pd.NA})
        # calculated_data DataFrame을 필요한 형식으로 변환하고, sum_of_usage를 DataFrame으로 변환
        calculated_data_df = pd.DataFrame(calculated_data)


        return calculated_data_df, sum_of_usage
    
    def outlier_processing_2(self, calculated_data_df):
        # 'usage_five' 기준으로 데이터 정렬
        # sorted_data_df = calculated_data_df.sort_values(by='usage_five')
        filtered_df = calculated_data_df.dropna(subset=['usage_five'])
        # 1사분위수와 3사분위수 계산
        q1 = filtered_df['usage_five'].quantile(0.25)
        q3 = filtered_df['usage_five'].quantile(0.75)
        # print(q1)
        # print(q3)
        # 평균과 표준편차 계산
        mean = filtered_df['usage_five'].mean()
        std_dev = filtered_df['usage_five'].std()

        # 중앙값 계산
        median = filtered_df['usage_five'].median()

        # IQR 계산
        iqr = q3 - q1
        upper_iqr = q3 + self.config['outlier']['very_large'] * iqr
        lower_iqr = q1 - self.config['outlier']['very_large'] * iqr

        # Three sigma Rule
        upper_sigma = mean + self.config['outlier']['very_large'] * std_dev
        lower_sigma = mean - self.config['outlier']['very_large'] * std_dev

        # Mean Absolute Deviation (MAD)
        mad = 1.4826 * abs(filtered_df['usage_five'] - median).median()
        upper_MAD = median + self.config['outlier']['very_large'] * mad
        lower_MAD = median - self.config['outlier']['very_large'] * mad

        # Skewed boxplot
        SIQRu = q3 - median
        SIQRl = median - q1
        upper_skewed = q3 + self.config['outlier']['very_large'] * SIQRu
        lower_skewed = q1 - self.config['outlier']['very_large'] * SIQRl

        # 상한값과 하한값을 결정합니다.
        upper = sorted([upper_iqr, upper_sigma, upper_MAD, upper_skewed])[-2]
        lower = sorted([lower_iqr, lower_sigma, lower_MAD, lower_skewed])[1]

        # 이상치 판단 조건
        conditions = (calculated_data_df['usage_five'] < lower) | (calculated_data_df['usage_five'] > upper)
        # 이상치를 None으로 대체
        calculated_data_df.loc[conditions, 'usage_five'] = pd.NA
        return calculated_data_df, upper
    
    def Isolation_Forest_Outlier_filter(self, calculated_data_df):
        # 'new_datetime'을 사용해 'EpochTime' 계산
        calculated_data_df['EpochTime'] = (calculated_data_df['new_datetime'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        calculated_data_df['EpochTime'] = calculated_data_df['EpochTime'].astype(float)
        
        # 'usage_five'에서 None 값을 가진 행은 제외하고 특성 선택
        features = calculated_data_df[['EpochTime', 'usage_five']].dropna()

        # 특성 표준화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Isolation Forest 모델 생성 및 학습
        model = IsolationForest(contamination=0.01)  # contamination 비율은 조정 가능
        model.fit(features_scaled)

        # 이상치 예측
        preds = model.predict(features_scaled)
        calculated_data_df.loc[features.index, 'Outlier'] = preds
        
        # 이상치(-1)를 None으로 대체
        calculated_data_df.loc[calculated_data_df['Outlier'] == -1, 'usage_five'] = pd.NA

        # 'EpochTime' 및 'Outlier' 열 제거 (필요하지 않은 경우)
        calculated_data_df.drop(columns=['EpochTime', 'Outlier'], inplace=True)

        return calculated_data_df, model, scaler
    
    def AutoEncoder_Outlier_filter(self, calculated_data_df):
        # `new_datetime`과 `new_date`는 이상치 탐지에 사용되지 않으므로 제외하고,
        # `usage_five` 열만을 특성으로 선택
        features = calculated_data_df[['usage_five']].dropna()  # NaN 값 제거

        # 특성 표준화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 학습 및 테스트 데이터 분리
        X_train, X_test = train_test_split(features_scaled, test_size=0.2, random_state=42)

        # AutoEncoder 모델 정의
        input_layer = Input(shape=(X_train.shape[1],))
        encoded = Dense(10, activation='relu')(input_layer)
        decoded = Dense(X_train.shape[1], activation='sigmoid')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        # 모델 학습
        autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test), verbose=0)

        # 이상치 탐지
        reconstructed_data = autoencoder.predict(features_scaled)
        mse = np.mean(np.power(features_scaled - reconstructed_data, 2), axis=1)
        threshold = np.percentile(mse, 99.95)  # 예시로 상위 0.05%를 이상치로 정의

        # 이상치 판단
        calculated_data_df['mse'] = pd.NA  # 모든 행에 mse 컬럼을 추가하고 초기값을 NaN으로 설정
        calculated_data_df.loc[features.index, 'mse'] = mse  # 계산된 mse 값 업데이트
        calculated_data_df['Outlier'] = 0  # 모든 행에 Outlier 컬럼을 추가하고 초기값을 0으로 설정
        calculated_data_df.loc[calculated_data_df['mse'] > threshold, 'Outlier'] = 1  # 이상치로 판단된 행의 Outlier 값을 1로 설정

        # 이상치를 None (np.nan)으로 대체
        calculated_data_df.loc[calculated_data_df['Outlier'] == 1, 'usage_five'] = pd.NA

        # 'mse' 및 'Outlier' 열 제거 (필요하지 않은 경우)
        calculated_data_df.drop(columns=['mse', 'Outlier'], inplace=True)

        return calculated_data_df, autoencoder, scaler, threshold
    
    
    def MCD_Outlier_filter(self, calculated_data_df):
        # 'usage_five' 열만을 특성으로 선택하고 NaN 값 제거
        features = calculated_data_df[['usage_five']].dropna()

        # 특성 표준화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if features_scaled.shape[0] < 1:
            # 입력 배열에 샘플이 없는 경우 처리
            print("Warning: Empty input array. Skipping outlier filtering.")
            return calculated_data_df, None, scaler, None

        # Minimum Covariant Determinant 모델 정의 및 학습
        mcd = MinCovDet(support_fraction=0.9)
        mcd.fit(features_scaled)

        # Mahalanobis 거리 계산
        mahalanobis_distance = mcd.mahalanobis(features_scaled)

        # 이상치 탐지 (예시로 상위 0.05%를 이상치로 정의)
        threshold = np.percentile(mahalanobis_distance, 99.95)

        # 이상치 판단
        calculated_data_df['mahalanobis_distance'] = np.nan  # 모든 행에 'mahalanobis_distance' 컬럼 추가
        calculated_data_df.loc[features.index, 'mahalanobis_distance'] = mahalanobis_distance  # 거리 값 업데이트
        calculated_data_df['Outlier'] = 0  # 모든 행에 'Outlier' 컬럼 추가 및 초기화
        calculated_data_df.loc[calculated_data_df['mahalanobis_distance'] > threshold, 'Outlier'] = 1

        # 이상치를 None (np.nan)으로 대체
        calculated_data_df.loc[calculated_data_df['Outlier'] == 1, 'usage_five'] = pd.NA

        # 'mahalanobis_distance' 및 'Outlier' 열 제거 (필요하지 않은 경우)
        calculated_data_df.drop(columns=['mahalanobis_distance', 'Outlier'], inplace=True)

        return calculated_data_df, mcd, scaler, threshold
    
    def OneClassSVM_Outlier_filter(self, calculated_data_df):
        # 'usage_five' 열만을 특성으로 선택하고 NaN 값 제거
        features = calculated_data_df[['usage_five']].dropna()

        # 특성 표준화
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # One-class SVM 모델 생성 및 학습
        model = OneClassSVM(nu=0.0005)  # nu는 이상치 비율을 나타냅니다. 적절히 조절해야 합니다.
        model.fit(features_scaled)

        # 이상치 예측
        preds = model.predict(features_scaled)

        # 예측 결과를 원본 데이터프레임에 할당
        calculated_data_df.loc[features.index, 'Outlier'] = preds

        # 이상치(-1)를 None으로 대체
        calculated_data_df.loc[calculated_data_df['Outlier'] == -1, 'usage_five'] = pd.NA

        # 'Outlier' 열 제거 (필요하지 않은 경우)
        calculated_data_df.drop(columns=['Outlier'], inplace=True)

        return calculated_data_df, model, scaler
    
    def outlier_processing(self, processed_outlier_list):
        # 모든 처리된 데이터프레임을 하나로 합치기
        combined_df = pd.concat(processed_outlier_list)
        
        # 각 'original_datetime'에 대해 'new_date'를 기준으로 데이터가 몇 번 나타났는지 계산
        # 여기서는 'original_datetime'과 'new_date'의 조합이 고유해야 하므로, 두 컬럼을 모두 사용
        combined_df['count'] = combined_df.groupby(['original_datetime', 'new_datetime'])['new_datetime'].transform('count')
        
        # 각 'original_datetime'에 대해 'count'가 1보다 큰 경우만 필터링하여 정상 데이터로 간주
        # 즉, 2개 이상의 이상치 판정을 받지 않은 데이터
        final_df = combined_df[combined_df['count'] > 1].drop_duplicates(['original_datetime', 'new_datetime'])
        
        # 필요한 컬럼만 선택하여 최종 데이터프레임 반환
        final_df = final_df[['new_datetime', 'usage_five']]
        
        return final_df
    
    def Calculate_Consumption_minute_new(self, final_df):

        # 'new_datetime'에서 시간 단위로 그룹화하기 위해 시간 키 생성
        final_df['hour_key'] = final_df['new_datetime'].dt.floor('H')
        
        # 시간 키('hour_key') 별로 사용량('usage_five') 합산
        hourly_usage = final_df.groupby('hour_key')['usage_five'].sum().reset_index(name='total_usage_hourly')

        return hourly_usage
    
    def fill_in_hour(self, hourly_usage):
        # 'hour_key' 컬럼을 datetime 타입으로 변환
        hourly_usage['hour_key'] = pd.to_datetime(hourly_usage['hour_key'])
        
        # 시작 시간과 종료 시간 결정
        start_date = hourly_usage['hour_key'].min()
        end_date = hourly_usage['hour_key'].max()
        
        # 시작 시간부터 종료 시간까지 모든 시간 생성
        all_hours = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # 모든 시간을 포함하는 데이터프레임 생성
        all_hours_df = pd.DataFrame(all_hours, columns=['hour_key'])
        
        # 원본 데이터와 모든 시간 데이터프레임 병합
        merged_df = pd.merge(all_hours_df, hourly_usage, on='hour_key', how='left')
        
        # 누락된 시간에 대한 'total_usage_hourly' 값을 None으로 설정
        merged_df['total_usage_hourly'].fillna(value=pd.NA, inplace=True)
        
        # 결과 데이터프레임 반환
        return merged_df
    
    def Interpolate_prophet (self, filled_data, total_usage):

        if total_usage is not None: 
            total_usage_df = pd.DataFrame(total_usage, columns=['hour_key', 'sum_of_usage'])
            total_usage_df['hour_key'] = pd.to_datetime(total_usage_df['hour_key'], format='%Y%m%d%H')  # 'ds' 열을 datetime 타입으로 변환

            # 원본 데이터셋과 total_usage 데이터 병합
            merged_df = pd.merge(filled_data, total_usage_df, on='hour_key', how='left')
            merged_df.rename(columns={'hour_key': 'ds', 'total_usage_hourly': 'y'}, inplace=True)
            # 누락된 데이터가 아닌 행만 필터링하여 모델 학습 데이터 준비
            train_data = merged_df[merged_df['y'].notna()].copy()

            # Prophet 모델 학습
            model = Prophet()
            model.fit(train_data[['ds', 'y']])

            for index, row in merged_df.iterrows():
                if pd.isna(row['y']) and not pd.isna(row['sum_of_usage']):
                    # 누락 시작 인덱스 식별
                    start_index = index
                    # 누락된 시간의 총 개수 계산 (다음 total_usage 값이 나타날 때까지)
                    missing_count = 1
                    while pd.isna(merged_df.iloc[start_index + missing_count]['y']):
                        missing_count += 1
                    
                    # 누락된 시간대에 대한 예측 수행
                    future = model.make_future_dataframe(periods=missing_count, freq='h', include_history=False)
                    forecast = model.predict(future)

                    forecast['yhat_adjusted'] = forecast['yhat'].apply(lambda x: max(x, 0))

                    # 예측된 합계와 실제 누적 사용량을 기반으로 조정 계수 계산
                    predicted_sum = forecast['yhat_adjusted'].sum()
                    
                    adjustment_factor = row['sum_of_usage'] / predicted_sum

                    adjusted_values = forecast['yhat_adjusted'] * adjustment_factor
                    for i, val in enumerate(adjusted_values):
                        merged_df.at[start_index + i, 'y'] = val

            merged_df.rename(columns={'ds':'hour_key', 'y':'total_usage_hourly'}, inplace=True)

            return merged_df[['hour_key', 'total_usage_hourly']]
            # merged_df['ds'] = merged_df['ds'].dt.strftime('%Y%m%d%H')  # 날짜 형식을 문자열로 변환
            # final_list = merged_df[['ds', 'y']].values.tolist()  # 결측치가 아닌 행만 선택하여 리스트로 변환

        elif total_usage is None:
            filled_data.rename(columns={'hour_key': 'ds', 'total_usage_hourly': 'y'}, inplace=True)
            train_data = filled_data[filled_data['y'].notna()].copy()
            model = Prophet()
            model.fit(train_data[['ds', 'y']])

            for index, row in filled_data.iterrows():
                if pd.isna(row['y']):
                    # 누락 시작 인덱스 식별
                    start_index = index
                    # 누락된 시간의 총 개수 계산 (다음 total_usage 값이 나타날 때까지)
                    missing_count = 1
                    while pd.isna(merged_df.iloc[start_index + missing_count]['y']):
                        missing_count += 1
                    
                    # 누락된 시간대에 대한 예측 수행
                    future = model.make_future_dataframe(periods=missing_count, freq='h', include_history=False)
                    forecast = model.predict(future)

                    forecast['yhat_adjusted'] = forecast['yhat'].apply(lambda x: max(x, 0))
                    for i, val in enumerate(forecast['yhat_adjusted']):
                        filled_data.at[start_index + i, 'y'] = val

            # merged_df['hour_key'] = merged_df['hour_key'].dt.strftime('%Y%m%d%H')  # 날짜 형식을 문자열로 변환
            # final_list = merged_df[['hour_key', 'total_usage_hourly']].values.tolist()  # 결측치가 아닌 행만 선택하여 리스트로 변환
            filled_data.rename(columns={'ds':'hour_key', 'y':'total_usage_hourly'}, inplace=True)
            return filled_data
    
    def map_season(self, month):
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8, 9]:
            return 'Summer'
        elif month == 10:
            return 'Autumn'
        else:
            return 'Winter'

    def make_season_usage_average(self, merged_df, selected_site_code):
        df_new = pd.DataFrame({
            'Hour': merged_df['hour_key'].dt.hour,
            'Month': merged_df['hour_key'].dt.month,
            'total_usage_hourly': merged_df['total_usage_hourly']
        })

        # 계절 컬럼 추가
        df_new['Season'] = df_new['Month'].apply(self.map_season)

        # 시간 및 계절별로 그룹화하여 평균 계산
        average_usage = df_new.groupby(['Season', 'Hour'])['total_usage_hourly'].mean().unstack(level=0)
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        average_usage = average_usage[season_order]

        average_usage_list = average_usage.values.T.tolist()

        directory = './season_average_usage/'
        file_path = f'{directory}{selected_site_code}_seasonal_hourly_average.csv'
        with open(file_path, 'w') as file:
            for season_data in average_usage_list:
                season_str = ','.join(map(str, season_data))
                file.write(season_str + '\n')
                
    # def transform_2D_to_1D(self, c_consumption_data, in_w_data, out_w_data):
    #     # 각 데이터를 numpy 배열로 변환합니다.
    #     cold_consumption_data = np.array(c_consumption_data).flatten()
    #     in_weather_data = np.array(in_w_data).flatten()
    #     out_weather_data = np.array(out_w_data).flatten()

        
    #     return in_weather_data, out_weather_data, cold_consumption_data
    
    def make_date_matched_data(self, cold, finedust, weather):

        finedust_processed = [(datetime.strptime(d[0], '%Y%m%d%H'), d[1], d[2]) for d in finedust]
        weather_processed = [(datetime.strptime(d[0], '%Y%m%d%H'), float(d[1]), float(d[2]), d[3]) for d in weather]
        # 최종 데이터 병합
        final_data = []
        for fd_date, fd_temp, fd_humid in finedust_processed:
            # weather와 매칭
            weather_match = next((w for w in weather_processed if w[0] == fd_date), None)
            # cold와 매칭
            # cold_match = cold[cold['hour_key'] == fd_date]['total_usage_hourly'].values
            cold_match = cold.loc[cold['hour_key'] == fd_date, 'total_usage_hourly'].values

            
            if weather_match and len(cold_match) > 0:
                final_data.append([
                    fd_date,  # 날짜
                    fd_temp,  # finedust 온도
                    fd_humid,  # finedust 습도
                    weather_match[1],  # weather 온도
                    weather_match[2],  # weather 습도
                    weather_match[3], 
                    cold_match[0]  # cold 사용량
                ])

        # 최종 데이터를 pandas DataFrame으로 변환
        final_df = pd.DataFrame(final_data, columns=['Date', 'Finedust_Temp', 'Finedust_Humid', 'Weather_Temp', 'Weather_Humid', 'Weather','Cold_Usage'])
        return final_df
    
    ##휴일이면 0, 휴일이 아니면 1
    
    def Is_holiday(self, final_df, holiday_data):
        holiday_dates = set(np.array(holiday_data).flatten())
        final_df['Is_Workday'] = final_df['Date'].apply(lambda x: 0 if x.strftime('%Y%m%d') in holiday_dates or x.weekday() >= 5 else 1)

        return final_df
        
    # def make_date_matched_data(self, finedust, weather, cold, all):
    #     # finedust의 날짜 형식을 datetime으로 변환
        
    #     finedust_dates = [datetime.strptime(d[0], '%Y%m%d%H') for d in finedust]
        
    #     # weather의 날짜 형식을 datetime으로 변환
    #     weather_dates = [datetime.strptime(d[0], '%Y-%m-%d %H:%M:%S') for d in weather.tolist()]
        
    #     # cold와 all은 이미 pd.datetime을 사용하므로, 직접 변환할 필요가 없음
    #     cold_dates = cold['datetime'].tolist()
    #     all_dates = all['datetime'].tolist()
        
    #     # 모든 데이터셋에서 일치하는 날짜 찾기
    #     matched_dates = set(finedust_dates) & set(weather_dates) & set(cold_dates) & set(all_dates)
        
    #     # 일치하는 날짜에 대한 데이터 추출 및 결합
    #     final_data = []
    #     for date in matched_dates:
    #         finedust_data = next(item for item in finedust if datetime.strptime(item[0], '%Y%m%d%H') == date)
    #         weather_data = next(item for item in weather.tolist() if datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S') == date)
    #         cold_data = cold.loc[cold['datetime'] == date, 'usage'].iloc[0]
    #         all_data = all.loc[all['datetime'] == date, 'usage'].iloc[0]
            
    #         final_data.append([date, finedust_data[1], finedust_data[2], weather_data[1], weather_data[2], cold_data, all_data])
        
    #     return final_data
    
    # def transform_2D_to_1D(self, c_consumption_data, in_w_data, out_w_data, d_consumption_data):
    #     # 각 데이터를 numpy 배열로 변환합니다.
    #     cold_consumption_data = np.array(c_consumption_data).flatten()
    #     in_weather_data = np.array(in_w_data).flatten()
    #     out_weather_data = np.array(out_w_data).flatten()
    #     daily_consumption_data = np.array(d_consumption_data).flatten()
        
    #     return in_weather_data, out_weather_data, cold_consumption_data, daily_consumption_data
