# 時系列予測モデルの実装
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

# 位置エンコーダ
# 時系列データの場合は相対エンコーディングを適用
class PositionEncoder(nn.Module):
    def __init__(self, dim_model, max_length_token=1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        position_encoder = torch.zeros(max_length_token, dim_model)
        position = torch.arange(0, max_length_token, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-torch.log(torch.tensor(max_length_token * 2)) / dim_model))
        # 交互にsinとcosを適用
        position_encoder[:, 0::2] = torch.sin(position * div_term)
        position_encoder[:, 1::2] = torch.cos(position * div_term)

        position_encoder = position_encoder.unsqueeze(0).transpose(0, 1)
        self.register_buffer('position_encoder', position_encoder)

    # 位置エンコーダを適用
    def forward(self, x):
        position_encord = x + self.position_encoder[:x.size(0), :]
        return self.dropout(position_encord)
    
# 時系列予測transformer
class TransformerTimeSeries(nn.Module):
    def __init__(self, size_feature=256, num_encoder_layers=2, num_head=8, dropout=0.1) -> None:
        super().__init__()
        self.src_mask = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.position_encoder = PositionEncoder(size_feature)

        # エンコーダ
        # マルチヘッド
        self.encoder_block = nn.TransformerEncoderLayer(
            d_model=size_feature,
            nhead=num_head,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_block,
            num_layers=num_encoder_layers
        )

        # デコーダ(今回はただの数値予測なので線形層)
        self.decoder = nn.Linear(size_feature, 1)
        # デバイス
        self.to(self.device)
        if os.path.exists('model.pth'):
            print('load model')
            self.load_state_dict(torch.load('model.pth'))

    # 重み初期化
    def init_weights(self):
        print('init weights')
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    # 未来情報をマスク
    def _generate_mask_future(self,sz):
        # 上△行列
        mask=(torch.triu(torch.ones(sz,sz))==1).transpose(0,1)
        # 0の部分を無効化
        mask=mask.float().masked_fill(mask==0,float('-inf')).masked_fill(mask==1,float(0.0))
        return mask
    
    # 順伝播
    def forward(self, x):
        # 未来情報マスク
        self.src_mask = self._generate_mask_future(len(x)).to(self.device)
        # 順伝播(encoder→decoder)
        x = self.position_encoder(x)
        x = self.encoder(x, self.src_mask)
        x = self.decoder(x)
        return x

# 過学習抑制
class EarlyStopping:
    def __init__(self,patience=5):
        self.patience=patience
        self.counter=0
        self.best_score=None
        self.early_stop=False
        self.val_loss_min=np.Inf

    def __call__(self,val_loss):
        score=(-val_loss)
        if self.best_score is None:
            self.best_score=score
        elif score<self.best_score:
            self.counter+=1
            if self.counter>=self.patience:
                self.early_stop=True
        else:
            self.best_score=score
            self.counter=0

# データを分割してデータ化
def get_batch(source, i, batch_size, observation_period_num=60):
    seq_len=min(batch_size, len(source)-1-i)
    data=source[i:i+seq_len]
    input=torch.stack(torch.stack([item[0] for item in data]).chunk(observation_period_num,1))
    target=torch.stack(torch.stack([item[1] for item in data]).chunk(observation_period_num,1))

    return input, target

# モデルの訓練
def train_model(model, dataset_train, dataset_eval):
    # 学習パラメータ
    lr = 0.0001
    epoch = 500
    size_batch = 64
    patience = 20

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)
    early_stopping = EarlyStopping(patience=patience)
    num_epochs = 100

    # ロスのhistory
    train_loss_list=[]
    eval_loss_list=[]

    for epoch in range(num_epochs):
        # 学習
        model.train()
        total_loss_train=0.0
        for batch, i in enumerate(range(0,len(dataset_train),size_batch)):
            x, y = get_batch(dataset_train, i, size_batch)
            x = x.to(model.device)
            y = y.to(model.device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            # 逆伝播
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
        scheduler.step()
        total_loss_train=total_loss_train/len(dataset_train)

        # 検証
        model.eval()
        total_loss_eval = 0.0
        for i in range(0,len(dataset_eval),size_batch):
            data,targets = get_batch(dataset_eval,i,size_batch)
            output = model(data)
            total_loss_eval += len(data[0])*criterion(output, targets).cpu().item()
        total_loss_eval = total_loss_eval/len(dataset_eval)

        # ロスの履歴を保存
        train_loss_list.append(total_loss_train)
        eval_loss_list.append(total_loss_eval)

        if epoch % 10 == 0:
            print(f'epoch: {epoch}, train_loss: {total_loss_train}, eval_loss: {total_loss_eval}')

        # 早期終了
        early_stopping(total_loss_eval)
        if early_stopping.early_stop:
            print(f'{epoch:3d}:epoch | {total_loss_train:5.7} : train loss | {total_loss_eval:5.7} : valid loss')
            print("Early Stop")
            break
        
        # モデルの保存
        if total_loss_eval <= min(eval_loss_list):
            torch.save(model.state_dict(), 'model.pth')

        # ロスの履歴を画像として保存
        plt.plot(train_loss_list, label='train')
        plt.plot(eval_loss_list, label='eval')
        plt.legend()
        plt.savefig('loss.png')


if __name__ == '__main__':
    from data_creater import create_dataset
    datasets_train, datasets_eval = create_dataset()
    for key, dataset_train in datasets_train.items():
        dataset_eval = datasets_eval[key]
        model = TransformerTimeSeries()
        model.init_weights()
        train_model(model, dataset_train, dataset_eval)