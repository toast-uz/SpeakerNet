'''
A machine learning for a speaker recognition:

Copyright(c) 2021 Tatsuzo Osawa (toast-uz)
All rights reserved. This program and the accompanying materials
are made available under the terms of the MIT License:
    https: // opensource.org/licenses/mit-license.php
'''

import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchaudio
from torchvision import transforms

torchaudio.set_audio_backend('sox_io')
MFCC_DIM = 24


# CommonVoiceをもとにtransformできるようにしたデータセット
# CommonVoiceはあらかじめダウンロードしておくこと
# https://commonvoice.mozilla.org/ja/datasets
class CommonVoiceDataset(Dataset):
    sample_rate = 16000

    def __init__(self, train=True, transform=None, split_rate=0.7):
        tsv = './CommonVoice/cv-corpus-5.1-2020-06-22/ja/validated.tsv'
        # データセットの一意性確認と正解ラベルの列挙
        import pandas as pd
        df = pd.read_table(tsv)
        assert not df.path.duplicated().any()
        self.classes = df.client_id.drop_duplicates().tolist()
        self.n_classes = len(self.classes)
        # データセットの準備
        self.transform = transform
        data_dirs = tsv.split('/')
        dataset = torchaudio.datasets.COMMONVOICE(
            '/'.join(data_dirs[:-4]), tsv=data_dirs[-1],
            url='japanese', version=data_dirs[-3])
        # データセットの分割
        n_train = int(len(dataset) * split_rate)
        n_val = len(dataset) - n_train
        torch.manual_seed(torch.initial_seed())  # 同じsplitを得るために必要
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
        self.dataset = train_dataset if train else val_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, sample_rate, dictionary = self.dataset[idx]
        # リサンプリングしておくと以降は共通sample_rateでtransformできる
        if sample_rate != self.sample_rate:
            x = torchaudio.transforms.Resample(sample_rate)(x)
        # 各種変換、MFCC等は外部でtransformとして記述する
        # ただし、推論とあわせるためにMFCCは先にすませておく（1〜24次元のみ抽出）
        x = torchaudio.transforms.MFCC(n_mfcc=MFCC_DIM+1)(x)[:, 1:]
        # 最終的にxのサイズを揃えること
        if self.transform:
            x = self.transform(x)
        # 特徴量:音声テンソル、正解ラベル:話者IDのインデックス
        return x, self.classes.index(dictionary['client_id'])


# 学習モデル
class SpeakerNet(nn.Module):
    def __init__(self, n_classes, n_hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm1d(MFCC_DIM),
            nn.Conv1d(MFCC_DIM, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )
        self.fc = nn.Sequential(
            nn.Linear(30*64, n_hidden),
            nn.BatchNorm1d(n_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(n_hidden, n_classes),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 最後の1次元に指定サイズにCropし、長さが足りない時はCircularPadする
# 音声データの時間方向の長さを揃えるために使うtransform部品
class CircularPad1dCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        n_repeat = self.size // x.size()[-1] + 1
        repeat_sizes = ((1,) * (x.dim() - 1)) + (n_repeat,)
        out = x.repeat(*repeat_sizes).clone()
        return out.narrow(-1, 0, self.size)


def SpeakerML(train_dataset=None, val_dataset=None, test_dataset=None, *,
              n_classes=None, n_hidden=256, n_epochs=15,
              load_pretrained_state=None, test_last_hidden_layer=False,
              show_progress=True, show_chart=False, save_state=False):
    '''
    前処理、学習、検証、推論を行う
    train_dataset: 学習用データセット
    val_dataset: 検証用データセット
    test_dataset: テスト用データセット（正解ラベル不要）
    n_classes: 分類クラス数（Noneならtrain_datasetから求める）
    n_epocs: 学習エポック数
    load_pretrained_state: 学習済ウエイトを使う場合の.pthファイルのパス
    test_last_hidden_layer: テストデータの推論結果に最終隠れ層を使う
    show_progress: エポックの学習状況をprintする
    show_chart: 結果をグラフ表示する
    save_state: test_acc > 0.9 の時のtest_loss最小値更新時のstateを保存
   　　　　　　　 （load_pretrained_stateで使う）
    返り値: テストデータの推論結果
    '''
    # モデルの準備
    if not n_classes:
        assert train_dataset, 'train_dataset or n_classes must be a valid.'
        n_classes = train_dataset.n_classes
    model = SpeakerNet(n_classes, n_hidden)
    if load_pretrained_state:
        model.load_state_dict(torch.load(load_pretrained_state))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # 前処理の定義
    train_transform = transforms.Compose([
        CircularPad1dCrop(800),
        transforms.RandomCrop((MFCC_DIM, random.randint(160, 320))),
        transforms.Resize((MFCC_DIM, 240)),
        lambda x: torch.squeeze(x, -3),
    ])
    val_transform = transforms.Compose([
        CircularPad1dCrop(240),
        lambda x: torch.squeeze(x, -3)
    ])
    test_transform = val_transform
    # 学習データ・テストデータの準備
    batch_size = 32
    train_dataloader = []
    if train_dataset:
        train_dataset.transform = train_transform
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)
    else:
        n_epochs = 0
    val_dataloader = []
    if val_dataset:
        val_dataset.transform = val_transform
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = []
    if test_dataset:
        test_dataset.transform = test_transform
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    # 学習
    losses = []
    accs = []
    val_losses = []
    val_accs = []
    for epoch in range(n_epochs):
        # 学習ループ
        running_loss = 0.0
        running_acc = 0.0
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            running_loss += loss.item()
            pred = torch.argmax(y_pred, dim=1)
            running_acc += torch.mean(pred.eq(y_train).float())
            optimizer.step()
        running_loss /= len(train_dataloader)
        running_acc /= len(train_dataloader)
        losses.append(running_loss)
        accs.append(running_acc)
        # 検証ループ
        val_running_loss = 0.0
        val_running_acc = 0.0
        for val_test in val_dataloader:
            if not(type(val_test) is list and len(val_test) == 2):
                break
            x_val, y_val = val_test
            y_pred = model(x_val)
            val_loss = criterion(y_pred, y_val)
            val_running_loss += val_loss.item()
            pred = torch.argmax(y_pred, dim=1)
            val_running_acc += torch.mean(pred.eq(y_val).float())
        val_running_loss /= len(val_dataloader)
        val_running_acc /= len(val_dataloader)
        can_save = (val_running_acc > 0.9 and
                    val_running_loss < min(val_losses))
        val_losses.append(val_running_loss)
        val_accs.append(val_running_acc)
        if show_progress:
            print(f'epoch:{epoch}, loss:{running_loss:.3f}, '
                  f'acc:{running_acc:.3f}, val_loss:{val_running_loss:.3f}, '
                  f'val_acc:{val_running_acc:.3f}, can_save:{can_save}')
        if save_state and can_save:   # あらかじめmodelフォルダを作っておくこと
            torch.save(model.state_dict(), f'model/0001-epoch{epoch:02}.pth')
    # グラフ描画
    if n_epochs > 0 and show_chart:
        fig, ax = plt.subplots(2)
        ax[0].plot(losses, label='train loss')
        ax[0].plot(val_losses, label='val loss')
        ax[0].legend()
        ax[1].plot(accs, label='train acc')
        ax[1].plot(val_accs, label='val acc')
        ax[1].legend()
        plt.show()
    # 推論
    if not test_dataset:
        return
    if test_last_hidden_layer:
        model.fc = model.fc[:-1]  # 最後の隠れ層を出力する
    y_preds = torch.Tensor()
    for test_data in test_dataloader:
        x_test = test_data[0] if type(test_data) is list else test_data
        y_pred = model.eval()(x_test)
        if not test_last_hidden_layer:
            y_pred = torch.argmax(y_pred, dim=1)
        y_preds = torch.cat([y_preds, y_pred])
    return y_preds.detach()


# 呼び出しサンプル
if __name__ == '__main__':
    train_dataset = CommonVoiceDataset(train=True)
    val_dataset = CommonVoiceDataset(train=False)
    result = SpeakerML(train_dataset, val_dataset, n_hidden=256, n_epochs=100,
                       show_chart=True, save_state=True)
