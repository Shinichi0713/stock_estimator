# 時系列データ取得
from pandas_datareader import data as wb
import yfinance as yfin
import torch
import numpy as np

# yahooファイナンスから株価の時系列データを取得
def download_dataset():
    yfin.pdr_override()

    # 株価コード
    # 株式分割のない銘柄を選択
    stock_codes = [
        '6367.T', # ダイキン工業
        '8001.T', # 伊藤忠商事
        '6501.T' # トヨタ自動車
    ]

    # 取得開始日
    start_date='2003-1-1'
    # 取得終了日
    end_date='2023-12-31'

    # データ読出し
    dataframes = {}
    for stock_code in stock_codes:
        df=wb.DataReader(stock_code,start=start_date,end=end_date)
        dataframes[stock_code] = df

    return dataframes


def create_dataset():
    # データ取得
    dataframes = download_dataset()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    ## 分割比率
    train_rate=0.7
    ## 時系列予測のための観測期間
    observation_period_num=60
    ## 予測期間
    predict_period_num=5

    

    # 時系列データ
    datasets_train = {}
    datasets_eval = {}
    for key, df in dataframes.items():
        # 正規化
        mean_list=df.mean().values
        std_list=df.std().values
        df=(df-mean_list)/std_list
        inout_data = []
        for i in range(len(df)-observation_period_num-predict_period_num):
            data=df.iloc[i:i+observation_period_num,4].values
            label=df.iloc[i+predict_period_num:i+observation_period_num+predict_period_num,4].values
            inout_data.append((data,label))
        inout_data=torch.FloatTensor(inout_data)

        dataset_train=inout_data[:int(np.shape(inout_data)[0]*train_rate)].to(device)
        dataset_eval=inout_data[int(np.shape(inout_data)[0]*train_rate):].to(device)
        datasets_train[key] = dataset_train
        datasets_eval[key] = dataset_eval

        print('train data：',np.shape(dataset_train)[0])
        print('valid data：',np.shape(dataset_eval)[0])
    
    return datasets_train, datasets_eval

if __name__ == '__main__':
    create_dataset()