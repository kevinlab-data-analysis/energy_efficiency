# import csv
import torch
import torch.nn as nn
import torch.optim as optim
# from datetime import datetime
from torch.utils.data import Dataset, DataLoader

class humidity_prediction_CNN(nn.Module):
    def __init__(self):
        super(humidity_prediction_CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv1(x)
        # x = F.relu(x)
        x = self.conv2(x)
        # x = F.relu(x)
        x = self.dropout(x) 
        x = x.view(x.size(0), -1)  #Flatten
        x = self.fc(x)
        return x

class humidity_prediction_dataset(Dataset):
    def __init__(self, data):
        inputs = []
        outputs = []
        for i in range(len(data) - 1):
            ## input : temp(T), hum(T), temp(T+1), output : hum(T+1)
            inputs.append([data.iloc[i]['Finedust_Temp'], data.iloc[i]['Finedust_Humid'], data.iloc[i + 1]['Finedust_Temp']])
            outputs.append([data.iloc[i + 1]['Finedust_Humid']])
            # inputs.append([float(data[i][1]), float(data[i][2]), float(data[i + 1][1])])
            # outputs.append([float(data[i + 1][2])])

        self.inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1).permute(0, 2, 1)
        self.outputs = torch.tensor(outputs, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            'input': self.inputs[idx],
            'output': self.outputs[idx]
        }
        return sample
    
class humidity_prediction:
    def __init__(self, data):
        self.data = data
        self.batch_size = 1
        self.epochs = 200

    def make_season_data(self):
        self.data['month'] = self.data['Date'].dt.month
        self.data['season'] = self.data['month'].apply(lambda x: 'summer' if 5 <= x <= 10 else ('winter'))
        summer_data = self.data[self.data['season'] == 'summer']
        winter_data = self.data[self.data['season'] == 'winter']
        return summer_data, winter_data
    
    def make_DataLoader(self, season_dataset):
        input_dataset = humidity_prediction_dataset(season_dataset)
        train_size = int(0.9 * len(input_dataset))  # 전체 데이터 중 90%를 훈련 데이터로 사용
        val_size = int(0.05 * len(input_dataset))
        test_size = len(input_dataset) - (train_size + val_size)
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(input_dataset, [train_size, val_size, test_size])

        train_DataLoader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_DataLoader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        test_DataLoader = DataLoader(test_dataset, batch_size=self.batch_size)
        return train_DataLoader, validation_DataLoader, test_DataLoader
    
    def humidity_prediction_train(self, train_DataLoader, validation_DataLoader, model_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = humidity_prediction_CNN().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        best_loss = 100.0
        patience = 0
        for epoch in range(self.epochs):
            patience += 1
            train_loss = 0.0
            for i, data in enumerate(train_DataLoader):
                inputs, targets = data['input'], data['output']
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                # 손실 계산
                loss = criterion(outputs, targets)  # 출력과 실제 값의 손실 계산
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
            average_train_loss = train_loss / len(train_DataLoader)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {average_train_loss}")

            model.eval()  # 모델을 평가 모드로 설정
            val_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(validation_DataLoader):
                    inputs, targets = data['input'], data['output']
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            average_val_loss = val_loss / len(validation_DataLoader)
            print(f"Epoch {epoch+1}/{self.epochs} - Validation Loss: {average_val_loss}")
            # writer.add_scalar("train_loss", average_train_loss, epoch)
            # writer.add_scalar("validation_loss", average_val_loss, epoch)

            # early stopping
            if 50 < patience:
                print('Train early stop')
                break
            
            # save best model
            if average_val_loss < best_loss:
                patience = 0
                best_loss = average_val_loss
                torch.save(model.state_dict(), model_path)
    
    def humidity_prediction_test(self, test_DataLoader, model_path):
        model = humidity_prediction_CNN()
        model.load_state_dict(torch.load(model_path))
        model.eval()
        criterion = nn.MSELoss()
        # criterion = nn.L1Loss()
        test_loss = 0.0
        y_pred = []
        with torch.no_grad():
            for i, data in enumerate(test_DataLoader):
                inputs, targets = data['input'], data['output']
                outputs = model(inputs)
                y_pred.append(outputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss / len(test_DataLoader)}")
