# 学習したモデルにより株価予測を行う
import pandas as pd
import torch, os
import numpy as np
import model
import matplotlib.pyplot as plt

def make_estimate(model):
    train_rate=0.7
    
    path_stock = os.path.dirname(__file__) + "/honda_stock_prices.csv"
    df_time=pd.read_csv(path_stock)

    # "Close"の列を時系列データとして抽出
    time_series_data = df_time["Close"]
    time_series_data = time_series_data.str.replace(",", "").astype(float)
    time_series_data = time_series_data[20:]
    # Normalization
    mean_list=time_series_data.mean()
    std_list=time_series_data.std()
    time_series_data=(time_series_data-mean_list)/std_list
    time_series_data = time_series_data
    # ウィンドウサイズ60で2次元配列に変換
    window_size = 60
    num_samples = len(time_series_data) - window_size + 1
    time_series_array = np.zeros((num_samples, window_size))

    for i in range(num_samples):
        time_series_array[i] = time_series_data[i:i + window_size].astype(float)

    # 最初の数行を表示
    inout_data_time=torch.FloatTensor([time_series_array])

    train_data_time=inout_data_time[:int(np.shape(inout_data_time)[0]*train_rate)].to(model.device)
    valid_data_time=inout_data_time[int(np.shape(inout_data_time)[0]*train_rate):].to(model.device)
    valid_data_time = valid_data_time.transpose(0, 1)
    return valid_data_time, time_series_data[int(np.shape(inout_data_time)[0]*train_rate):].to_numpy()


def predict():
    trans_estimator = model.TransformerTimeSeries()
    trans_estimator.eval()
    result=torch.Tensor(0)
    valid_data_time, wave_input = make_estimate(trans_estimator)

    with torch.no_grad():
        for i in range(0,len(valid_data_time)-1):
            data_validate = {}
            input = torch.unsqueeze(valid_data_time[i].transpose(0,1), 2)
            output = trans_estimator(input)
            result = torch.cat((result, output[-1].view(-1).cpu()),0)
        # numpyに変換
        data_validate['predict'] = result.cpu().detach().numpy()
        data_validate['actual'] = wave_input[0:result.shape[0]]

    plt.plot(data_validate['predict'],color='red',linewidth=1.0, label='predict')
    plt.plot(data_validate['actual'],color='blue',linewidth=1.0, label='actual')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    predict()