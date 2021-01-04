'''
A speaker diarization utilities:

Copyright(c) 2021 Tatsuzo Osawa (toast-uz)
All rights reserved. This program and the accompanying materials
are made available under the terms of the MIT License:
    https: // opensource.org/licenses/mit-license.php
'''

import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn import cluster
import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision
from speakernet import SpeakerML, MFCC_DIM
MODEL_PATH = 'data/model/cnn1d_20210102.pth'


class Speech(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        if 'window' in kwargs:
            del kwargs['window']
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs):
        self.window = 1.0
        if 'window' in kwargs:
            self.window = kwargs['window']

    def __getitem__(self, idx):
        return Speech(self.to_tensor()[idx], window=self.window)

    @property
    def duration(self):
        return self.window * self.size()[-1]

    def to_tensor(self):
        return torch.Tensor(self).clone()

    def clone(self, *, window=None):
        if not window:
            window = self.window
        return Speech(self.to_tensor(), window=window)

    @classmethod
    def read(cls, filename):  # mp3, mp4も利用可能（要pyav）
        if filename.split('.')[-1] in ['wav', 'mp3']:
            torchaudio.set_audio_backend('sox_io')
            x, sample_rate = torchaudio.load(filename)
        else:
            _, x, info = torchvision.io.read_video(filename)
            sample_rate = info['audio_fps']
        x = x.mean(0, keepdim=True)   # モノラル変換
        return Speech(x, window=1/sample_rate)

    def transform(self, transform_func):
        return Speech(transform_func(self.to_tensor()), window=self.window)

    def resample(self, window=1/16000):
        if self.window == window:
            return self
        x = self.clone(window=window)
        f1, f2 = 1 / self.window, 1 / window
        if window > 1.0:
            f1, f2 = f1 * window, f2 * window
        return x.transform(torchaudio.transforms.Resample(f1, f2))

    def amplitude_to_db(self):
        x = self.transform(torchaudio.transforms.AmplitudeToDB(top_db=80))
        return x.transform(lambda x: x + 80)

    def mfcc(self, *, log_mels=False, window=None, hop=None):
        '''
        windowのデフォルトはself.window * 400
        hopのデフォルトはwindow/2 = self.windwo * 200
        出力結果には次元0を含む
        出力結果のwindow = hop
        '''
        if not window:
            window = self.window * 400
        if not hop:
            hop = window / 2
        n_fft = int(window / self.window)
        hop_length = int(hop / self.window)
        x = self.clone()
        x = torchaudio.transforms.MFCC(
            n_mfcc=MFCC_DIM+1, log_mels=log_mels,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length})(x)
        return Speech(x, window=hop)

    def normalize(self):
        x = self.to_tensor()
        x = (x - x.mean(-1, )) / (x.std(-1) + 1e-08)
        return Speech(x, window=self.window)

    def vad2(self, *, window=3.0, n_search_chunks=3):
        assert self.dim() == 2, 'Input must be a wave.'
        chunk_size = int(window / self.window)
        chunks = list(self.to_tensor().split(chunk_size, dim=-1))
        labels = []
        n_vad_chunks = n_search_chunks
        for i in range(len(chunks) - n_search_chunks + 1):
            if n_vad_chunks == n_search_chunks:
                speech_ = torch.cat(chunks[i: i+n_search_chunks], dim=-1)
                speech_vad = torchaudio.transforms.Vad(
                    1 / self.window)(speech_)
                n_vad_chunks = speech_vad.size()[-1] // chunk_size + 1
            labels.append(0 if n_vad_chunks < n_search_chunks else 1)
            n_vad_chunks = min(n_vad_chunks + 1, n_search_chunks)
            print(f'{i}: {speech_.size()} -> {speech_vad.size()}'
                  f' = {labels[-1]} * {n_search_chunks-n_vad_chunks+1}')
        return Speech(torch.Tensor(labels).unsqueeze(0), window=window)

    def vad3(self, *, window=3.0):
        assert self.dim() == 3, 'Input must be MFCC before normalized.'
        _, x = self.clone().resample(window).to_tx()
        clustering = cluster.KMeans(n_clusters=2).fit(x.T)
        labels = clustering.labels_
        # mfccの大きさを比較して、無音側を判断し、クラスタ0にする
        if (np.abs(x[:, labels == 0]).sum() >
                np.abs(x[:, labels == 1]).sum()):
            labels = 1 - labels
        return Speech(torch.Tensor(labels).unsqueeze(0), window=window)

    def vad(self, *, window=3.0, filter_window=1.0, threshold=0.2):
        assert self.dim() == 3 and self.size()[-2] == MFCC_DIM+1, \
            'Input must be MFCC with power coefficient 0.'
        x = self[0, 0, :].normalize().numpy()
#        x[x < 0] = 0
        x = scipy.ndimage.gaussian_filter(x, filter_window / self.window)
#        x = (x > threshold).astype(int)
        x = Speech(x, window=self.window).resample(window).numpy()
#        x = (x > 0.5).astype(int)
        return Speech(torch.Tensor(x).unsqueeze(0), window=window)

    def diarized(self, vad, *, n_speakers=None):
        # n_speaker: 話者数, Noneなら自動判別

        # x_vectorの計算
        assert self.dim() == 3 and self.size()[-2] == MFCC_DIM+1, \
            'Input must be MFCC with power coefficient 0.'
        x = self.transform(lambda x: x[:, 1:])
        dataset = ChunkedMFCCDataset(x, vad)
        x_vectors = SpeakerML(test_dataset=dataset, n_classes=170,
                              load_pretrained_state=MODEL_PATH,
                              test_last_hidden_layer=True)
        # コサイン類似度の計算
        cos_sim = Speech(x_vectors).cos_sim()
        if not n_speakers:  # 話者数自動判別
            # eigenvalues = torch.from_numpy(cos_sim).eig()[0][:, 0]
            # eigengaps = eigenvalues[:-1] / eigenvalues[1:]

            n_speakers = 5
        # クラスタリング
        labels = cluster.SpectralClustering(
            n_clusters=n_speakers, affinity='precomputed').fit_predict(cos_sim)
        # クラスタラベルを置換して出現順をラベル昇順（1〜）にする
        _, unique_indexes = np.unique(labels, return_index=True)
        for i, unique_index in enumerate(sorted(unique_indexes)):
            labels[labels == labels[unique_index]] = n_speakers + i
        labels = labels - n_speakers + 1
        # 無音ラベル:0とあわせて、labels*1をvad.windowの時間間隔に変換する
        out = []
        for label, start, length in zip(
                labels, dataset.speeches_start, dataset.speeches_length):
            out.extend([0] * (start - len(out)))
            out.extend([label] * length)
        return Speech(torch.Tensor(out).unsqueeze(0),
                      window=dataset.chunk_window)

    # ユーティリティ関数群

    def to_tx(self):
        '''
        返り値
        x: -1次元が時間軸、-2次元が特徴量軸である、2次元のndarray
        t: xの時刻だが、最後はxの最終項の終了時刻であり、xの時間軸より1つ要素が多い
        '''
        t = np.arange(self.size()[-1] + 1) * self.window
        x = self.numpy().copy()
        while x.ndim > 2:
            assert x.shape[0] == 1
            x = x.squeeze(0)
        return t, x

    @classmethod
    def tx_slice(cls, t, x, pos: tuple = None):
        if not pos:
            pos = (t[0], t[-1])
        is_keep = (pos[0] < t) & (t < pos[1])
        is_keep = is_keep | np.append(is_keep[1:], False)
        x_slice = x.T[is_keep[:-1]].T
        is_keep = is_keep | np.append(False, is_keep[:-1])
        return t[is_keep] - pos[0], x_slice

    @classmethod
    def tx_gap(cls, t, x):   # xの変化点を抽出
        assert x.shape[0] == 1
        x_is_diff = np.append(True, x[0, :-1] != x[0, 1:])
        return t[np.append(x_is_diff, True)], x[:, x_is_diff]

    # PyTorchのCosineSimilarityは、構造上、メモリを大量に使うので、独自実装
    def cos_sim(self, eps: float = 1e-08):
        x = self.numpy().copy().squeeze()
        d = x @ x.T
        norm = (x * x).sum(axis=1, keepdims=True) ** .5
        norm = np.concatenate(
            [norm, np.ones(norm.shape)*eps], 1).max(axis=1).reshape(-1, 1)
        return d / norm / norm.T

    # 描画関数群

    def _draw_subplots(self, ax, *, chart, pos=None):
        t, x = self.to_tx()
        n_class = int(x.max()) + 1
        t, x = Speech.tx_slice(t, x, pos)
        y = np.arange(len(x) + 1)
        if chart == 'wave':
            if x.shape[0] == 1:
                x = x.squeeze(0)
            ax.fill_between(t[:-1], x, 0)
        elif chart == 'heat':
            ax.pcolormesh(t, y, x, shading='flat', cmap='jet')
        elif chart in ['class', 'class2']:
            assert x.shape[0] == 1
            t_gap, x_gap = Speech.tx_gap(t, x)
            ax.pcolormesh(t_gap, y, x_gap, shading='flat', cmap='rainbow',
                          vmin=0, vmax=n_class-1)
            if chart == 'class2':
                ax.vlines(t_gap, 0, y.max(), colors='black')

    def show(self, *, chart, width=None, window=None):
        '''
        chart: 'wave'  : 強度波形
               'heat'  : ヒートマップ
               'class' : 分類
               'class2': 分類+境界線
        '''
        assert chart in ['wave', 'heat', 'class', 'class2'], \
            'Parameter chart is invalid.'
        # グラフの解像度の調整
        x = self.clone()
        if not window:
            window = max(0.1, self.window)
        if window != self.window:
            x = x.resample(window)
        # グラフ描画を複数行にするための幅と行数を求める
        if not width:
            width = self.duration
        plt_num = math.ceil(self.duration / width)  # math.floor代替
        # 描画エリアの準備
        fig, ax = plt.subplots(plt_num)
        if plt_num == 1:
            ax = (ax,)  # グラフ1つの時はaxがスカラーなので調整
        for i in range(plt_num):
            x._draw_subplots(ax[i], chart=chart,
                             pos=(width*i, width*(i+1)))
            # 描画領域の整形
            ax[i].set_xlim(0, width)
            ax[i].tick_params(labelleft=False)
            ax[i].set_ylabel(f'{width * i // 60}')  # 分で表示
            if i == 0:  # 最初のグラフだけ表題をつけて、全体の表題に見せる
                ax[i].set_title('Speech viewer')
                ax[i].set_ylabel(f'{width * i // 60}(m)')
            if i == plt_num - 1:  # 最後のグラフだけx軸ラベルをつける
                ax[i].set_xlabel('Time(s)')
            else:
                ax[i].tick_params(labelbottom=False)
        plt.show()


class ChunkedMFCCDataset(Dataset):
    '''
    mfcc: Speech型のMFCCデータをvad.window間隔でチャンク化し、無音部分で分断し、
    最大max_n_chunk個数分のチャンクをつなげたTensorをデータセット化する
    '''

    def __init__(self, mfcc, vad, *, transform=None, max_n_chunk=1):
        self.transform = transform
        self.chunk_window = vad.window  # チャンク時間
        self.speeches = []              # 分割したMFCCデータ
        self.speeches_start = []        # 分割開始位置/単位時間
        self.speeches_length = []       # 分割長/単位時間
        chunks = list(mfcc.to_tensor().split(
            int(self.chunk_window / mfcc.window), dim=-1))
        vad_labels = vad.to_tensor().squeeze().tolist()
        chunks.append(None)   # forループの最後の処理を完遂するための調整
        vad_labels.append(0)  # 同上
        n_chunks = 0
        speech_ = torch.Tensor()
        for i, (chunk, vad_label) in enumerate(zip(chunks, vad_labels)):
            if ((vad_label == 0 and n_chunks > 0) or
                    (n_chunks == max_n_chunk)):
                self.speeches.append(speech_)
                self.speeches_length.append(n_chunks)
                n_chunks = 0
                speech_ = torch.Tensor()
            if vad_label == 0:
                continue
            n_chunks += 1
            speech_ = torch.cat([speech_, chunk], dim=-1)
            if n_chunks == 1:
                self.speeches_start.append(i)

    def __len__(self):
        return len(self.speeches)

    def __getitem__(self, idx):
        x = self.speeches[idx]
        if self.transform:
            x = self.transform(x)
        return x


# 呼び出しサンプル
if __name__ == '__main__':
    x = Speech.read('data/meetings/meeting1.mp3')
    x = x.resample().mfcc(log_mels=True)
    vad = x.vad(threshold=0.2)
    vad.show(chart='wave', width=300)
#    x.diarized(vad, n_speakers=4).show(chart='class', width=300)
