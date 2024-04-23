import os
import re
import yaml
import time
import shutil
import warnings
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from Humidity_prediction import humidity_prediction
from Cold_usage_prediction_tools import prediction_tools
from data_preprocessing_tools_np import lbems_data_tools

warnings.filterwarnings('ignore', message='pandas only supports SQLAlchemy connectable')

with open('./config_auto_train.yaml', 'r') as file:
    config = yaml.safe_load(file)


def auto_train(selected_site_code):
    # start_time = time.time()
    selected_site_code = selected_site_code
    db = config['default']['db']
    host = config['default']['host']
    lbems = lbems_data_tools(config)
    query = selected_site_code + '_config'

    
    finedust_query = config[query]['meter_finedust']
    finedust_raw_data = lbems.Read_Data_in_DB_new(host, db, finedust_query, [1, 2], 'float64')
    except_none_finedust = lbems.Except_None_Data(finedust_raw_data)
    first_info_finedust = lbems.Make_Finedust_Data(except_none_finedust)
    print('finish findust')


    # from csv file(weather information) -> output [date, temperature, humidity]
    weather_query = config[query]['meter_weather']
    weather_raw = lbems.Read_Data_in_DB_new(host, db, weather_query, list(range(1, 49)), 'float64')
    except_extracted_weather = lbems.Except_None_Data(weather_raw)
    except_extracted_weather = lbems.make_weather_datasets(except_extracted_weather)
    print('finish weather')

    # meter_electric_cold -> output [date, cold consumption per hour]
    find_cold_query = config[query]['meter_electric_cold']
    cold_result = lbems.Read_Data_in_DB_new(host, db, find_cold_query, [2], 'float64')
    except_extracted_cold = lbems.Except_None_Data(cold_result)
    divided_cold = lbems.Cold_Sensor_Divide(except_extracted_cold)
    cold_sum = lbems.Calculate_Consumption_cold(divided_cold)
    filled_cold_sum = lbems.fill_in_hour(cold_sum)
    filled_interpolate_prophet_cold_sum = lbems.Interpolate_prophet(filled_cold_sum, None)
    print('finish cold')

    lbems.make_season_usage_average(filled_interpolate_prophet_cold_sum, selected_site_code)

    # ## 최종 input data
    # final_data = lbems.make_date_matched_data(filled_interpolate_prophet_cold_sum, first_info_finedust, except_extracted_weather)

    # ## make prediction model
    # now_date = datetime.now().strftime("%Y%m%d%H%M")
    # site_code = query.split('_')[0]

    # humidity_prediction_model_summer_save_path = config[query]['humidity_summer_path']
    # humidity_prediction_model_winter_save_path = config[query]['humidity_winter_path']
    # Hum = humidity_prediction(final_data)
    # summer_data, winter_data = Hum.make_season_data()

    # ## move pre model
    # pt = prediction_tools(final_data)
    # model_list_dir = './model_list/' + selected_site_code
    # model_dir = './weights/' + selected_site_code
    # extensions = ['.pth', '.pkl', '.zip']

    # # 확장자 목록
    # if not os.path.exists(model_list_dir):
    #     os.makedirs(model_list_dir, exist_ok=True)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir, exist_ok=True)

    # pt.move_model_file(model_list_dir,model_dir,extensions)

    # ## train humidity prediction model with summer data
    # train_DataLoader, validation_DataLoader, test_DataLoader = Hum.make_DataLoader(summer_data)
    # Hum.humidity_prediction_train(train_DataLoader, validation_DataLoader, humidity_prediction_model_summer_save_path)
    # # Hum.humidity_prediction_test(test_DataLoader, humidity_prediction_model_summer_save_path)

    # ## train humidity prediction model with winter data
    # train_DataLoader, validation_DataLoader, test_DataLoader = Hum.make_DataLoader(winter_data)
    # Hum.humidity_prediction_train(train_DataLoader, validation_DataLoader, humidity_prediction_model_winter_save_path)
    # # Hum.humidity_prediction_test(test_DataLoader, humidity_prediction_model_winter_save_path)

    # XGB_model_save_path = config[query]['XGB_save_path']
    # LGBM_model_save_path = config[query]['LGBM_save_path']
    # Tabnet_model_save_path = config[query]['Tabnet_save_path']
    # meta_model_save_path = config[query]['meta_model_save_path']

    # final_data['Weather']  = final_data['Weather'].astype('category')
    # label_encoder = LabelEncoder()
    # final_data['Weather'] = label_encoder.fit_transform(final_data['Weather'])

    # X_train, X_val, X_test, y_train, y_val, y_test = pt.Dataset()
    # XGB_pred, rmse, mae, mape, XGB_model = pt.train_XGB(X_train, X_val, X_test, y_train, y_val, y_test, XGB_model_save_path)
    # LGBM_pred, rmse, mae, mape, LGBM_model = pt.train_LGBM(X_train, X_val, X_test, y_train, y_val, y_test, LGBM_model_save_path)
    # Tabnet_pred, rmse, mae, mape, Tabnet_model = pt.train_Tabnet(X_train, X_val, X_test, y_train, y_val, y_test, Tabnet_model_save_path)
    # rmse_meta, meta_model = pt.stacking(XGB_model, LGBM_model, Tabnet_model, X_val, X_test, y_val, y_test, meta_model_save_path)

    # print('RMSE with Stacking:', rmse_meta)
    # end_time = time.time()
    # print('total time : ', end_time - start_time)