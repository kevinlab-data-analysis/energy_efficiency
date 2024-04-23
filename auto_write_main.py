import os
import time
import yaml
import warnings
from datetime import datetime, timedelta
from usage_reduce_scenario import reduction_scenario, run_main
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def auto_write(site_code):
    start_time = time.time()
    with open('./config_auto_write.yaml', 'r') as file:
        config = yaml.safe_load(file)

    rs = reduction_scenario(config)
    rm = run_main()

    now_time = datetime.now()
    now_time_date = now_time.strftime("%Y%m%d")
    now_time_hour = now_time.strftime("%H")
    one_hour_ago = now_time - timedelta(hours=1)
    one_hour_ago_hour = one_hour_ago.strftime("%H")

    temp_before_query = 'temperature_' + one_hour_ago_hour
    hum_before_query = 'humidity_' + one_hour_ago_hour
    weather_sun_before_query = 'weather_' + one_hour_ago_hour

    temp_query = 'temperature_' + now_time_hour
    hum_query = 'humidity_' + now_time_hour
    weather_sun_query = 'weather_' + now_time_hour

    input_data = []
    site_code = site_code
    site_parameter = site_code + '_parameter'
    sensor_list = config[site_parameter]['sensor_list']

    if now_time_hour != '00':
        weather_query = config[site_parameter]['weather_query'].format(temp_before = temp_before_query, temp = temp_query, hum_before = hum_before_query, hum = hum_query, weather_before = weather_sun_before_query, weather = weather_sun_query, date = now_time_date)
        weather_data = rm.Read_Data_in_DB(config['DB_parameter']['host'], config['DB_parameter']['user'], config['DB_parameter']['passwd'], config['DB_parameter']['db'], weather_query)[0]
    else:
        weather_ago_query = config[site_parameter]['weather_query_ago'].format(temp = temp_before_query, hum = hum_before_query, weather = weather_sun_before_query, date = one_hour_ago)
        weather_now_query = config[site_parameter]['weather_query_now'].format(temp = temp_query,hum = hum_query,weather = weather_sun_query, date = now_time_date)

        weather_ago_data = rm.Read_Data_in_DB(config['DB_parameter']['host'], config['DB_parameter']['user'], config['DB_parameter']['passwd'], config['DB_parameter']['db'], weather_ago_query)
        weather_now_data = rm.Read_Data_in_DB(config['DB_parameter']['host'], config['DB_parameter']['user'], config['DB_parameter']['passwd'], config['DB_parameter']['db'], weather_now_query)
        weather_data = (weather_ago_data[0][0],)
        for i in range(1, len(weather_ago_data[0])):
            weather_data += (weather_ago_data[0][i],)  # 첫 번째 데이터셋에서 항목 추가
            if i < len(weather_now_data[0]):  # 두 번째 데이터셋에서 항목이 있을 경우만 추가
                weather_data += (weather_now_data[0][i-1],)  # 두 번째 데이터셋에서 항목 추가


    for sensor_sn in sensor_list:
        input_data_per_sensor = []

        input_data_per_sensor.append(now_time.strftime("%Y%m%d%H%M%S"))

        findust_query = config[site_parameter]['finedust_query'].format(sensor_sn=sensor_sn)
        findust_data = rm.Read_Data_in_DB(config['DB_parameter']['host'], config['DB_parameter']['user'], config['DB_parameter']['passwd'], config['DB_parameter']['db'], findust_query)
        input_data_per_sensor = rm.append_findust(input_data_per_sensor, findust_data)

        input_data_per_sensor = rm.append_weather(input_data_per_sensor, weather_data, now_time_hour)

        input_data_per_sensor.append(sensor_sn)
        # input_data = date, in_temp, in_hum, out_temp, out_hum, com_code_pk, sensorsn
        input_data.append(input_data_per_sensor)

    summer_humidity_model_path = config[site_parameter]['summer_humidity_model_path']
    winter_humidity_model_path = config[site_parameter]['winter_humidity_model_path']
    XGB_prediction_model_path = config[site_parameter]['XGB_model_path']
    LGBM_prediction_model_path = config[site_parameter]['LGBM_model_path']
    Tabnet_prediction_model_path = config[site_parameter]['Tabnet_model_path']
    usage_prediction_model_path = config[site_parameter]['usage_prediction_model_path']
    model_path_list = [summer_humidity_model_path, winter_humidity_model_path, 
                       XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path, usage_prediction_model_path]
    
    ## 계절별 시간별 평균 사용량(shape = (4, 24))
    average_usage_path = f'./season_average_usage/{site_code}_seasonal_hourly_average.csv'
    average_usage_df = pd.read_csv(average_usage_path, header=None)

    # DataFrame의 값을 float 리스트의 리스트로 변환
    average_usage_list = average_usage_df.values.tolist()

    # 리스트의 각 원소를 float으로 변환
    average_usage_list = [[float(i) for i in row] for row in average_usage_list]

    # average_usage_list = [[1108.89, 1115.56, 1158.89, 1146.67, 1197.78, 1360.0, 2327.78, 5441.11, 
    #                     6237.78, 5471.11, 4413.64, 4071.11, 3627.78, 3311.24, 2845.45, 2424.44, 
    #                     2278.65, 2073.03, 1187.64, 1293.33, 1260.0, 1240.0, 1144.44, 1134.44], 
    #                     [2193.53, 1873.6, 1769.77, 1780.93, 1739.43, 1769.67, 3180.1, 7647.37, 
    #                         11217.5, 12113.7, 11652.58, 12006.5, 12769.7, 11602.89, 11288.07, 10218.62, 
    #                         9178.53, 7833.31, 5015.17, 3860.73, 3324.43, 2800.23, 2520.7, 2304.63], 
    #                         [1266.14, 1017.29, 967.29, 922.0, 988.86, 950.29, 1189.71, 1423.57, 2420.29, 
    #                         2502.29, 2311.29, 1930.71, 1952.86, 2018.81, 1827.14, 1715.86, 1399.14, 1284.43, 
    #                         1189.71, 1160.0, 1209.86, 1248.0, 1425.29, 1240.0], 
    #                         [4468.87, 4414.53, 4173.16, 4401.83, 4492.9, 4529.96, 8725.23, 20078.8, 26694.36, 
    #                         24234.02, 19997.77, 16523.92, 14773.62, 13066.91, 11082.93, 9688.07, 8938.73, 8317.42, 
    #                         6138.37, 5629.27, 5477.63, 5412.04, 5004.81, 4733.59]]

    for input in input_data:
        date = input[0]
        in_temp = input[1]
        in_hum = input[2]
        out_temp = input[3]
        out_hum = input[4]
        weather = input[5]
        complex_code_pk = input[6]
        zone = input[7]

        pmv_error_code = '0'
        if 10 <= in_temp <= 40: 
            result_list, prediction_error_code = rs.scenario(date, in_temp, in_hum, out_temp, out_hum, weather, 
                                                          model_path_list, average_usage_list)
            rec_temp = float(result_list[0])
            rec_hum = float(result_list[1])
            usages = float(result_list[3])
            reduced_usage = float(result_list[2])
            now_PMV = float(result_list[4])
            rec_PMV = float(result_list[5])
            now_time = datetime.now()
            reg_time = now_time.strftime("%Y%m%d%H%M%S")
        else:
            rec_temp = None
            rec_hum = None
            usages = None
            reduced_usage = None
            now_PMV = None
            rec_PMV = None
            now_time = datetime.now()
            reg_time = now_time.strftime("%Y%m%d%H%M%S")
            pmv_error_code = '1'
            prediction_error_code = '0'

        print(usages, reduced_usage)
        rec_temp = round(rec_temp * 2) / 2
        error_code = prediction_error_code + pmv_error_code
        insert_query = "INSERT INTO pmv_info (date, error_code, complex_code_pk, home_dong_pk, home_ho_pk, zone, in_temp, in_hum, out_temp, out_hum, rec_temp, rec_hum, usages, reduced_usage, now_PMV, rec_PMV, reg_time) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        insert_data = (date, error_code, complex_code_pk, None, None, zone, in_temp, in_hum, out_temp, out_hum, rec_temp, rec_hum, usages, reduced_usage, now_PMV, rec_PMV, reg_time)
        rm.Write_Data_in_DB(config['DB_parameter']['host'], config['DB_parameter']['user'], config['DB_parameter']['passwd'], config['DB_parameter']['db'], query=insert_query, data=insert_data)
        end_time = time.time()
        print(site_code, end_time - start_time)
