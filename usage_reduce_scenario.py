import yaml
import math
import torch
import joblib
import pymysql
import numpy as np
import torch.nn as nn
from pythermalcomfort.models import pmv_ppd
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetRegressor

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

class reduction_scenario:
    def __init__(self, config):
        self.config = config

    def define_season(self, month):
        if 3 <= month <= 5:
            season = 'spring'
            I_cl = 0.825
        elif 6 <= month <= 9:
            season = 'summer'
            I_cl = 0.79
        elif month == 10:
            season = 'fall'
            I_cl = 0.825
        else:
            season = 'winter'
            I_cl = 1
        return season, I_cl

    def calculate_PMV(self, in_temp, in_hum, I_cl):
        T_a = in_temp
        RH = in_hum
        I_cl = I_cl
        M = self.config['PMV_parameter']['M']
        W = self.config['PMV_parameter']['W']
        v_ar = self.config['PMV_parameter']['v_ar']
        T_r = T_a
        PMV_PPD = pmv_ppd(tdb = T_a, tr = T_r, vr = v_ar, rh = RH, met = M, clo = I_cl, standard="ASHRAE")
        now_PMV = PMV_PPD['pmv']
        return now_PMV

    def search_optimal_PMV(self, now_PMV, season, month, in_temp, in_hum, I_cl, summer_humidity_model_path, winter_humidity_model_path):
        T_a = in_temp
        RH = in_hum

        if math.isnan(now_PMV):
            return T_a, now_PMV
        else:
            if season == 'summer':
                if now_PMV < self.config['PMV_parameter']['target_PMV']:
                    target_pmv = self.config['PMV_parameter']['target_PMV']
                    step = (self.config['PMV_parameter']['step'])
                else:
                    target_pmv = now_PMV
                    rec_temperature = T_a
                    current_pmv = now_PMV

            elif season == 'winter':
                if -(self.config['PMV_parameter']['target_PMV']) < now_PMV:
                    target_pmv = -(self.config['PMV_parameter']['target_PMV'])
                    step = -(self.config['PMV_parameter']['step'])
                else:
                    target_pmv = now_PMV
                    rec_temperature = T_a
                    current_pmv = now_PMV

            else:
                if -0.5 < now_PMV <= 0:
                    target_pmv = -(self.config['PMV_parameter']['target_PMV'])
                    step = -(self.config['PMV_parameter']['step'])
                elif 0 < now_PMV < 0.5:
                    target_pmv = self.config['PMV_parameter']['target_PMV']
                    step = (self.config['PMV_parameter']['step'])
                else:
                    target_pmv = now_PMV
                    rec_temperature = T_a
                    current_pmv = now_PMV

            cnt = 0
            humidity_prediction_model = SimpleCNN()

            if 5 <= month <= 10:
                model_path = summer_humidity_model_path
            else:
                model_path = winter_humidity_model_path

            humidity_prediction_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            humidity_prediction_model.eval()
            init_RH = RH
            # print(season)
            while True:
                cnt += 1
                if cnt != 1:
                    input = [T_a_t_1, RH, T_a]
                    # print('input : ', input)
                    input = torch.tensor(input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                    # print(input, input.shape)
                    predicted_humidity = humidity_prediction_model(input)
                    RH = predicted_humidity.item()
                current_pmv = self.calculate_PMV(T_a, RH, I_cl)
                
                if abs(current_pmv - target_pmv) <= self.config['PMV_parameter']['PMV_gap']:
                    break
                
                T_a_t_1 = T_a
                T_a += step
                T_a = round(T_a, 2)

            rec_PMV = current_pmv
            rec_temperature = T_a
            rec_humidity = RH
        return rec_PMV, rec_temperature, rec_humidity

    def usage_check_process(self, usage, reduced_usage, temp_gap, h_time, season, average_usage_list):
        prediction_error_code = '1'
        if (usage <= 0) or (usage > 0 and reduced_usage <= 0) or (reduced_usage > usage):

            if season == 'spring':
                usage = average_usage_list[0][h_time]
                reduced_usage = usage - (temp_gap * 55.7)
                if reduced_usage < 0:
                    reduced_usage = usage * 0.9

            elif season == 'summer':
                usage = average_usage_list[1][h_time]
                reduced_usage = usage - (temp_gap * 55.7)
                if reduced_usage < 0:
                    reduced_usage = usage * 0.9

            elif season == 'fall':
                usage = average_usage_list[2][h_time]
                reduced_usage = usage - (temp_gap * 55.7)
                if reduced_usage < 0:
                    reduced_usage = usage * 0.9
            else:
                usage = average_usage_list[3][h_time]
                reduced_usage = usage - (temp_gap * 55.7)
                if reduced_usage < 0:
                    reduced_usage = usage * 0.9
        
        return usage, reduced_usage, prediction_error_code

    def make_meta_model_input(self, input, XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path):
        Tabnet_model = TabNetRegressor(
            input_dim = 8, 
            output_dim = 1, 
            n_d = 4, 
            n_a = 8, 
            n_steps = 3, 
            gamma = 1.2,
            n_independent = 2, 
            n_shared = 2, 
            optimizer_fn = torch.optim.Adam,
            optimizer_params = dict(lr=2e-2), 
            scheduler_params = {"step_size":50, "gamma":0.5},
            scheduler_fn = torch.optim.lr_scheduler.StepLR, 
            mask_type = 'entmax', 
            lambda_sparse = 1e-3, 
            seed = 42
            )

        XGB_model = joblib.load(XGB_prediction_model_path)
        LGBM_model = joblib.load(LGBM_prediction_model_path)
        Tabnet_model.load_model(Tabnet_prediction_model_path)

        XGB_y_pred = XGB_model.predict(input)
        LGBM_y_pred = LGBM_model.predict(input)
        input = np.array(input)
        Tabnet_y_pred = Tabnet_model.predict(input)

        XGB_y_pred = XGB_y_pred.reshape(-1, 1)
        LGBM_y_pred = LGBM_y_pred.reshape(-1, 1)

        meta_input = np.hstack([np.hstack([XGB_y_pred, Tabnet_y_pred]), LGBM_y_pred])

        return meta_input


    def predict_consumption(self, h_time, season, rec_temperature, rec_humidity, in_temp, in_hum, out_temp, out_hum, weather, 
                            XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path, 
                            usage_prediction_model_path, average_usage_list):
        
        usage_prediction_model = joblib.load(usage_prediction_model_path)
        prediction_error_code = '0'
        ## already in_temp is too high or too low, so no recammand
        if rec_temperature == in_temp:
            input_X = [[in_temp, in_hum, out_temp, out_hum, in_temp, in_hum, out_temp, out_hum, weather]]
            input_X = self.make_meta_model_input(input_X, XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path)
            # input_X = [[float(element) for element in input_X[0]]]
            usage = usage_prediction_model.predict(input_X)
            if usage <= 0:
                prediction_error_code = '1'
                if season == 'spring':
                    usage = average_usage_list[0][h_time]
                elif season == 'summer':
                    usage = average_usage_list[1][h_time]
                elif season == 'fall':
                    usage = average_usage_list[2][h_time]
                else:
                    usage = average_usage_list[3][h_time]
            return usage, usage, prediction_error_code
        
        else:
            temp_gap = round(abs(rec_temperature - in_temp), 2)

            usage_input = [[in_temp, in_hum, out_temp, out_hum, in_temp, in_hum, out_temp, out_hum, weather]]
            # usage_input = [[float(element) for element in usage_input[0]]]

            rec_usage_input = [[in_temp, in_hum, out_temp, out_hum, round(rec_temperature, 2), round(rec_humidity, 2), out_temp, out_hum, weather]]
            # rec_usage_input = [[float(element) for element in rec_usage_input[0]]]

            usage_input = self.make_meta_model_input(usage_input, XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path)
            rec_usage_input = self.make_meta_model_input(rec_usage_input, XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path)

            usage = usage_prediction_model.predict(usage_input)
            reduced_usage = usage_prediction_model.predict(rec_usage_input)

            usage, reduced_usage, prediction_error_code = self.usage_check_process(usage, reduced_usage, temp_gap, h_time, season, average_usage_list)

            return usage, reduced_usage, prediction_error_code

    ## 위의 함수 실행시키는 전체 main
    def scenario(self, date, in_temp, in_hum, out_temp, out_hum, weather, 
                model_path_list, average_usage_list):

        month = int(date[4] + date[5])
        h_time = int(date[8] + date[9]) 
        in_temp = float(in_temp)
        in_hum = float(in_hum)
        out_temp = float(out_temp)
        out_hum = float(out_hum)
        summer_humidity_model_path = model_path_list[0]
        winter_humidity_model_path = model_path_list[1]
        XGB_prediction_model_path = model_path_list[2]
        LGBM_prediction_model_path = model_path_list[3]
        Tabnet_prediction_model_path = model_path_list[4]
        usage_prediction_model_path = model_path_list[5]

        ## 계절, pmv에 들어가는 의복 계수 결정
        season, I_cl = self.define_season(month)

        ## 내부 온습도 기반의 현재 pmv값 도출
        now_PMV = self.calculate_PMV(in_temp, in_hum, I_cl)

        ## 기준 pmv까지 조정했을때의 온습도 도출(제안 온습도)
        rec_PMV, rec_temperature, rec_humidity = self.search_optimal_PMV(now_PMV, season, month, in_temp, in_hum, I_cl, summer_humidity_model_path, winter_humidity_model_path)

        ## 제안 온습도로 조정했을때의 에너지 사용양 예측
        # usage, reduced_usage = predict_consumption(rec_temperature, rec_humidity, in_temp, in_hum, out_temp, out_hum)
        # usage, reduced_usage = predict_consumption_ver2(hour_usage, rec_temperature, rec_humidity, in_temp, in_hum, out_temp, out_hum)
        # usage, reduced_usage = predict_consumption_ver3(rec_temperature, rec_humidity, in_temp, in_hum, out_temp, out_hum)
        usage, reduced_usage, prediction_error_code = self.predict_consumption(h_time, season, rec_temperature, rec_humidity, in_temp, in_hum, out_temp, out_hum, weather, 
                                                                        XGB_prediction_model_path, LGBM_prediction_model_path, Tabnet_prediction_model_path, 
                                                                        usage_prediction_model_path, average_usage_list)

        result_list = [rec_temperature, rec_humidity, reduced_usage, usage, now_PMV, rec_PMV]
        return result_list, prediction_error_code
    
class run_main:
    def Read_Data_in_DB(self, host, user, passwd, db, query):
        conn = pymysql.connect(
            host = host,
            user = user,
            passwd = passwd,
            db = db
        )

        curs = conn.cursor()
        curs.execute(query)
        total_result = curs.fetchall()
        conn.close()

        return total_result

    def Write_Data_in_DB(self, host, user, passwd, db, query, data):
        conn = pymysql.connect(
            host = host,
            user = user,
            passwd = passwd,
            db = db
        )

        curs = conn.cursor()
        curs.execute(query, data)
        conn.commit()
        conn.close()

    def append_findust(self, input_data, findust_data):
        if findust_data[0][7] is None:
            not_none_count = 0
            average_temp = 0
            for i in range (len(findust_data)):
                if findust_data[i][7] is None:
                    continue 
                else:
                    not_none_count += 1
                    average_temp += float(findust_data[i][7])
            input_data.append(average_temp/not_none_count)
        else:
            input_data.append(findust_data[0][7])

        if findust_data[0][8] is None:
            not_none_count = 0
            average_hum = 0
            for i in range (len(findust_data)):
                if findust_data[i][8] is None:
                    continue 
                else:
                    not_none_count += 1
                    average_hum += float(findust_data[i][8])
            input_data.append(average_hum/not_none_count)

        else:
            input_data.append(findust_data[0][8])

        return input_data


    def weather_encoding(self, weather):

        weather_to_string = {
            '01': 'clear_sky',
            '02': 'few_clouds',
            '03': 'scattered_clouds',
            '04': 'broken_clouds',
            '09': 'shower_rain',
            '10': 'rain',
            '11': 'thunderstorm',
            '13': 'snow',
            '50': 'mist'
        }
        
        weather_string = weather_to_string.get(weather[:-1], None)  # 매핑되지 않은 코드는 "None" 처리

        if weather_string == None:
            return weather_string
        else:
            kind_of_weather = list(weather_to_string.values())
            label_encoder = LabelEncoder()
            label_encoder.fit(kind_of_weather)
            encoded_weather = label_encoder.transform([weather_string])[0]
            return encoded_weather

    def append_weather(self, input_data, weather_data, now_time_hour):
        complex_code_pk = weather_data[0]

        # out temperature
        if weather_data[2] is None:
            input_data.append(float(weather_data[1]))    
        else:
            input_data.append(float(weather_data[2]))

        # out humidity
        if weather_data[4] is None:
            input_data.append(float(weather_data[3]))
        else:
            input_data.append(float(weather_data[4]))

        # weather
        if weather_data[6] is None:
            weather = weather_data[5]
        else:
            weather = weather_data[6]
            
        encoded_weather = self.weather_encoding(weather)
        input_data.append(encoded_weather)

        input_data.append(complex_code_pk)
        return input_data
