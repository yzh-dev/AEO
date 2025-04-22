from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio
import csv

target_all = [0, 1, 2, 3, 4, 5, 6, 7] #EPIC-KITCHENS数据集中的动作类别，模型训练时见过的类别
target_all_hac = [0, 1, 2, 3, 4, 5] #HAC数据集中的动作类别，只包含6个动作类别。模型未见过的类别，用于测试模型对未知类别的识别能力

""" take,0
put,1
open,2
close,3
wash,4
cut,5
mix,6
pour,7 """

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split() for line in lines]
    paths, labels = zip(*data)
    return paths, labels

def get_spectrogram_piece(samples, start_time, end_time, duration, samplerate, training=False):
    start1 = start_time / duration * len(samples)
    end1 = end_time / duration * len(samples)
    start1 = int(np.round(start1))
    end1 = int(np.round(end1))
    samples = samples[start1:end1]

    resamples = samples[:160000]
    if len(resamples) == 0:
        resamples = np.zeros((160000))
    while len(resamples) < 160000:
        resamples = np.tile(resamples, 10)[:160000]

    resamples[resamples > 1.] = 1.
    resamples[resamples < -1.] = -1.
    frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
    spectrogram = np.log(spectrogram + 1e-7)

    mean = np.mean(spectrogram)
    std = np.std(spectrogram)
    spectrogram = np.divide(spectrogram - mean, std + 1e-9)

    interval = 9
    if training is True:
        noise = np.random.uniform(-0.05, 0.05, spectrogram.shape)
        spectrogram = spectrogram + noise
        start1 = np.random.choice(256 - interval, (1,))[0]
        spectrogram[start1:(start1 + interval), :] = 0

    return spectrogram

class EPICDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='train', domain=['D1'], modality='rgb', cfg=None, cfg_flow=None, sample_dur=10, use_video=True, use_audio=True, datapath='/path/to/EPIC-KITCHENS/', datapath_open=''):
        self.base_path = datapath#数据集路径:D:/ML/Dataset/EPIC_KITCHENS/
        self.base_path_open = datapath_open#开放集数据集路径:D:/ML/Dataset/HAC/
        self.split = split
        self.modality = modality
        self.interval = 9
        self.sample_dur = sample_dur
        self.use_video = use_video
        self.use_audio = use_audio
        self.video_list = []
        self.prefix_list = []
        self.label_list = []

        # build the data pipeline
        if split == 'train':
            if self.use_video:
                train_pipeline = cfg.data.train.pipeline
                self.pipeline = Compose(train_pipeline)
        else:#AEO项目是TTA实验，因此split='test'
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)

        # 疑问：模型在D2域的train数据集上训练了么？
        data1 = []#保存EPIC-KITCHENS数据集中，用户给定的target domain的train和test数据
        splits = []
        for dom in domain:#domain=['D2']，用户给定的target domain
            for spl in ['train', 'test']:#读取EPIC-KITCHENS数据集的train和test数据
                train_file = pd.read_pickle('./splits/'+dom+"_"+spl+".pkl")#读取当前的划分数据

                for _, line in train_file.iterrows():
                    image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                            line['stop_timestamp']]
                    labels = line['verb_class']
                    if int(labels) in target_all:
                        data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
                        splits.append(spl)
        # HAC数据集会出现新的类别
        for dom in ['human', 'animal', 'cartoon']:#遍历HAC三种不同的视频来源，同时获取了其train和test数据
            prefix = dom + '/'
            with open("./splits/HAC_%s_only_%s.csv" % (split, dom)) as f:#test数据
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if int(row[1]) in target_all_hac:#只保留标签在 target_all_hac 列表中的视频。即只选择6个指定类别的动作视频
                        self.video_list.append(row[0])#视频文件名
                        self.prefix_list.append(prefix)#视频来源：'human', 'animal', 'cartoon'
                        self.label_list.append(row[1])#视频标签
            # 这里的train数据参与训练模型参数了么？
            with open("./splits/HAC_train_only_%s.csv" % (dom)) as f:#train数据，组织为列表形式，包含了'human', 'animal', 'cartoon'的数据
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if int(row[1]) in target_all_hac:
                        self.video_list.append(row[0])
                        self.prefix_list.append(prefix)
                        self.label_list.append(row[1])

        self.samples = data1#保存EPIC-KITCHENS数据集中，用户给定的target domain的train和test数据
        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.splits = splits#EPIC视频数据，包含"train"和"test"
        print("len(self.video_list): ", len(self.video_list))

    def __getitem__(self, index):
        video_path = self.base_path +'rgb/'+self.splits[index] + '/'+self.samples[index][0]#EPIC视频文件夹名
        # ------------------------------------------------------------------------------------------------
        # 给定index，获取EPIC中视频数据
        filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')#文件名模板
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))#开始帧
        data = dict(
            frame_dir=video_path,
            total_frames=int(self.samples[index][2] - self.samples[index][1]),#帧数
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality,
            fmt='img')
        data = self.pipeline(data,fmt='img')#返回EPIC数据集的视频,来自"train"或"test"

        # ------------------------------------------------------------------------------------------------
        # 获取HAC中视频数据，可能来自'human', 'animal', 'cartoon'
        # 如果超出了HAC的索引范围，处理到合法范围内
        index_open = index
        if index >= len(self.video_list):#len(self.video_list)=2881.  index包含对EPIC某个domain的样本量，其中，D1=1415+401，D2样本量为2281+694,D3=3530+903
            index_open = index - len(self.video_list)#
        video_file_open = self.base_path_open + self.prefix_list[index_open] +'videos/' + self.video_list[index_open]
        vid_open = iio.imread(video_file_open, plugin="pyav")#读取视频：[帧数，HWC]
        frame_num_open = vid_open.shape[0]#视频帧数
        start_frame_open = 0#视频开始帧
        end_frame_open = frame_num_open-1#视频结束帧索引
        start_index_open = self.cfg.data.val.get('start_index', start_frame_open)#
        data_open = dict(
            # frame_dir=video_path,
            frame_dir=video_file_open,#指定HAC视频路径
            # total_frames=end_frame_open - start_frame_open,
            total_frames=frame_num_open,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index_open,#
            video=vid_open,
            frame_num=frame_num_open,
            filename_tmpl=filename_tmpl,
            modality=modality,
            fmt='video')
        data_open, frame_inds_open = self.pipeline(data_open,fmt='video')#data_open代表预处理并提取的帧数据，frame_inds_open包含提取后的帧索引。含32帧
       #data_open["imgs"]: torch.Size([1, 3, 32, 224, 224])  
        # ------------------------------------------------------------------------------------------------
        # EPIC数据集中的音频数据
        audio_path = self.base_path + 'audio/' + self.splits[index] + '/' + self.samples[index][0] + '.wav'
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate
        # EPIC数据集中的开始和结束时间需要手动计算
        fr_sec = self.samples[index][3].split(':')#开始时间：时:分:秒
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1
        stop_sec = self.samples[index][4].split(':')#结束时间：时:分:秒
        hour1 = float(stop_sec[0])
        minu1 = float(stop_sec[1])
        sec1 = float(stop_sec[2])
        stop_sec = (hour1 * 60 + minu1) * 60 + sec1

        # start1 = fr_sec / duration * len(samples)
        # end1 = stop_sec / duration * len(samples)
        # start1 = int(np.round(start1))
        # end1 = int(np.round(end1))
        # samples = samples[start1:end1]

        # resamples = samples[:160000]
        # while len(resamples) < 160000:
        #     resamples = np.tile(resamples, 10)[:160000]

        # resamples[resamples > 1.] = 1.
        # resamples[resamples < -1.] = -1.
        # frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        # spectrogram = np.log(spectrogram + 1e-7)

        # mean = np.mean(spectrogram)
        # std = np.std(spectrogram)
        # spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        spectrogram = get_spectrogram_piece(samples,fr_sec,stop_sec,duration,samplerate)

        # ------------------------------------------------------------------------------------------------
        # HAC数据集中的音频数据，从0开始的，因此可以直接计算
        audio_path_open = self.base_path_open + self.prefix_list[index_open] + 'audio/' + self.video_list[index_open][:-4] + '.wav'
        start_time = frame_inds_open[0] / 24.0# 这里除以24是因为视频的帧率（Frame Rate）是24帧/秒（fps）。这是一个标准的视频帧率
        end_time = frame_inds_open[-1] / 24.0
        samples, samplerate = sf.read(audio_path_open)
        duration = len(samples) / samplerate
        spectrogram_open = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate)

        label1 = self.samples[index][-1]
        label1_open = int(8)#HAC的标签被硬编码为8，目的是将所有HAC数据统一标记为未知类别（8）

        return data, spectrogram, label1, data_open, spectrogram_open, label1_open

    def __len__(self):
        # return max(len(self.samples), len(self.samples_open))
        return len(self.samples)#返回保存EPIC-KITCHENS数据集中，用户给定的target domain的train和test数据样本和

