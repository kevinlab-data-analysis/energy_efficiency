import os
import re
import yaml
import torch
import shutil
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import timedelta
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from lightgbm import early_stopping
from sklearn.linear_model import LinearRegression
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

class prediction_tools:
    def __init__(self, data):
        self.data = data

    def Dataset(self):
        X = []
        Y = []
        self.data['Date'] = pd.to_datetime(self.data['Date'])

        for i in range(len(self.data) - 1):
            time_step = self.data.iloc[i + 1].iloc[0] - self.data.iloc[i].iloc[0]
            if time_step == timedelta(hours=1):
            # if time_step == timedelta(minutes = 10):
                if self.data.iloc[i + 1].iloc[5] != '':
                    ## (T) 내부, 외부 온습도, (T+1)내부, 외부 온습도, 1시간동안의 사용량
                    # X.append([self.data.iloc[i].iloc[1], self.data.iloc[i].iloc[2], self.data.iloc[i].iloc[3], 
                    #           self.data.iloc[i].iloc[4], self.data.iloc[i + 1].iloc[1], self.data.iloc[i + 1].iloc[2], 
                    #           self.data.iloc[i + 1].iloc[3], self.data.iloc[i + 1].iloc[4]])

                    # Y.append(float(self.data.iloc[i + 1].iloc[5]))

                    # 날씨, 휴일 추가
                    X.append([self.data.iloc[i].iloc[1], self.data.iloc[i].iloc[2], self.data.iloc[i].iloc[3], 
                              self.data.iloc[i].iloc[4], self.data.iloc[i].iloc[5], self.data.iloc[i + 1].iloc[1], 
                              self.data.iloc[i + 1].iloc[2], self.data.iloc[i + 1].iloc[3], self.data.iloc[i + 1].iloc[4]])
                    
                    Y.append(float(self.data.iloc[i + 1].iloc[6]))


        X_train, X_test_val, y_train, y_test_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_XGB(self, X_train, X_val, X_test, y_train, y_val, y_test, XGB_model_save_path):
        model = xgb.XGBRegressor(
            # general parameter
            # n_jobs=N, ## number of CPU thread using train
            booster='gbtree',
            verbosity=0,
            objective='reg:squarederror', ##MSE
            # objective='reg:pseudohubererror',
            #boost parameter
            learning_rate = 0.05,
            n_estimators = 500,
            max_depth = 9,
            min_child_weight = 1,
            gamma = 0,
            subsample = 0.7,
            colsample_bytree = 0.7,
            reg_lambda = 1,
            reg_alpha = 1
            )

        ## actual train
        model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)], 
        eval_metric='rmse',
        early_stopping_rounds=50,
        )

        ## model save
        model_dir = os.path.dirname(XGB_model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, XGB_model_save_path)

        ## test
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        return y_pred, rmse, mae, mape, model
    
    def train_LGBM(self, X_train, X_val, X_test, y_train, y_val, y_test, LGBM_model_save_path):
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        y_train = np.array(y_train)
        y_val = np.array(y_val)
        y_test = np.array(y_test)

        model = LGBMRegressor(
            # general parameter
            # device = 'gpu',
            verbosity=0,
            objective='regression',

            #boost parameter
            colsample_bytree = 1.0,
            learning_rate = 0.1,
            max_depth = -1,
            n_estimators = 500,
            num_leaves = 26,
            subsample = 0.8,
            reg_lambda = 0.0,
            reg_alpha = 1.0
            )

        ## actual train
        model.fit(
        X_train, 
        y_train,
        eval_set=[(X_val, y_val)], 
        eval_metric='mae',
        callbacks=[early_stopping(stopping_rounds=100, verbose=True)]
        )

        ## save model
        model_dir = os.path.dirname(LGBM_model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, LGBM_model_save_path)

        # test
        y_pred = model.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False) ## squared = True면 mse, fales면 rmse
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        return y_pred, rmse, mae, mape, model
    
    def train_Tabnet(self, X_train, X_val, X_test, y_train, y_val, y_test, Tabnet_model_save_path):
        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        y_train = np.array(y_train).reshape(-1, 1)
        y_val = np.array(y_val).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

        model = TabNetRegressor(
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
        
        model.fit(
            X_train = X_train, 
            y_train = y_train, 
            eval_set = [(X_val, y_val)], 
            eval_name = ['val'], 
            eval_metric = ['rmse'], 
            max_epochs = 200, 
            patience = 20, 
            batch_size = 512, 
            virtual_batch_size = 128, 
            num_workers = 0, 
            weights = np.ones(len(y_train)), 
            drop_last = False
            )
        
        ## save model
        model_dir = os.path.dirname(Tabnet_model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        model.save_model(Tabnet_model_save_path)

        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False) ## squared = True면 mse, fales면 rmse
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)

        return y_pred, rmse, mae, mape, model
    
    def stacking(self, XGB_model, LGBM_model, Tabnet_model, X_val, X_test, y_val, y_test, meta_model_save_path):
        ## meta model train
        XGB_y_pred = XGB_model.predict(X_val)
        LGBM_y_pred = LGBM_model.predict(X_val)
        X_val = np.array(X_val)
        Tabnet_y_pred = Tabnet_model.predict(X_val)

        XGB_y_pred = XGB_y_pred.reshape(-1, 1)
        LGBM_y_pred = LGBM_y_pred.reshape(-1, 1)

        meta_X_train = np.hstack([np.hstack([XGB_y_pred, Tabnet_y_pred]), LGBM_y_pred])
        meta_model = LinearRegression()
        meta_model.fit(meta_X_train, y_val)

        ## save model
        model_dir = os.path.dirname(meta_model_save_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        joblib.dump(meta_model, meta_model_save_path)

        ## meta model test
        XGB_y_pred_test = XGB_model.predict(X_test)
        LGBM_y_pred_test = LGBM_model.predict(X_test)
        X_test = np.array(X_test)
        Tabnet_y_pred_test = Tabnet_model.predict(X_test)

        XGB_y_pred_test = XGB_y_pred_test.reshape(-1, 1)
        LGBM_y_pred_test = LGBM_y_pred_test.reshape(-1, 1)

        meta_X_test = np.hstack([np.hstack([XGB_y_pred_test, Tabnet_y_pred_test]), LGBM_y_pred_test])
        y_pred_meta = meta_model.predict(meta_X_test)
        rmse_meta = mean_squared_error(y_test, y_pred_meta, squared=False)

        return rmse_meta, meta_model


    def make_graph(self, y_pred, y_test, now_date, site_code):
        plt.figure(figsize=(10,5))
        plt.plot(y_test[:100], label='Actual', color='green')
        plt.plot(y_pred[:100], label='Predicted', color='red')
        plt.title('Comparison of Actual and Predicted Values')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        figure_path = './v2/train_tools/graph/Tabnet_' + now_date + '_' + site_code + '.png'
        plt.savefig(figure_path)


    def get_next_version_number(self, model_base_name, model_list_dir):
        version_numbers = []
        for file in os.listdir(model_list_dir):
            if file.startswith(model_base_name):
                parts = re.split('[_\.]', file)  # 파일명을 _와 .을 기준으로 분리
                for part in parts:
                    if part.startswith('v') and part[1:].isdigit():
                        version_numbers.append(int(part[1:]))
                        break
        return max(version_numbers) + 1 if version_numbers else 1

    def move_model_file(self, model_list_dir, model_dir, extensions):
        # 모든 파일에 대해 처리
        for file_name in os.listdir(model_dir):
            for ext in extensions:
                if file_name.endswith(ext):
                    model_base_name = file_name[:-len(ext)]  # 확장자 제외한 모델 기본 이름
                    next_version = self.get_next_version_number(model_base_name, model_list_dir)
                    new_name = f"{model_base_name}_v{next_version}{ext}"
                    shutil.move(os.path.join(model_dir, file_name), os.path.join(model_list_dir, new_name))
                    break  # 현재 파일에 대한 처리 완료
