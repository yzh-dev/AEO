from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio
import csv

target_all = [0, 1, 2, 3, 4, 5, 6, 7]

target_all_hac = [0, 1, 2, 3, 4, 5]

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
        self.base_path = datapath
        self.base_path_open = datapath_open
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
        else:
            if self.use_video:
                val_pipeline = cfg.data.val.pipeline
                self.pipeline = Compose(val_pipeline)

        data1 = []
        splits = []
        for dom in domain:
            for spl in ['train', 'test']:
                train_file = pd.read_pickle(self.base_path + 'MM-SADA_Domain_Adaptation_Splits/'+dom+"_"+spl+".pkl")

                for _, line in train_file.iterrows():
                    image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                            line['stop_timestamp']]
                    labels = line['verb_class']
                    if int(labels) in target_all:

                        data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
                        splits.append(spl)

        for dom in ['human', 'animal', 'cartoon']:
            if dom == 'animal':
                prefix = 'ActorShift/'
            elif dom == 'human':
                prefix = 'kinetics600/'
            else:
                prefix = 'cartoon/'
            #prefix = dom + '/'
            with open(self.base_path_open + "HAC_Splits/HAC_%s_only_%s.csv" % (split, dom)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if int(row[1]) in target_all_hac:
                        self.video_list.append(row[0])
                        self.prefix_list.append(prefix)
                        self.label_list.append(row[1])

            with open(self.base_path_open + "HAC_Splits/HAC_train_only_%s.csv" % (dom)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    if int(row[1]) in target_all_hac:
                        self.video_list.append(row[0])
                        self.prefix_list.append(prefix)
                        self.label_list.append(row[1])

        self.samples = data1
        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.splits = splits
        print("len(self.video_list): ", len(self.video_list))

    def __getitem__(self, index):
        video_path = self.base_path +'rgb/'+self.splits[index] + '/'+self.samples[index][0]
        
        index_open = index
        if index >= len(self.video_list):
            index_open = index - len(self.video_list)
        
        filename_tmpl = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        modality = self.cfg.data.train.get('modality', 'RGB')
        start_index = self.cfg.data.train.get('start_index', int(self.samples[index][1]))
        data = dict(
            frame_dir=video_path,
            total_frames=int(self.samples[index][2] - self.samples[index][1]),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            filename_tmpl=filename_tmpl,
            modality=modality,
            fmt='img')
        data = self.pipeline(data,fmt='img')

        video_file_open = self.base_path_open + self.prefix_list[index_open] +'videos/' + self.video_list[index_open]
        vid_open = iio.imread(video_file_open, plugin="pyav")

        frame_num_open = vid_open.shape[0]
        start_frame_open = 0
        end_frame_open = frame_num_open-1
        start_index_open = self.cfg.data.val.get('start_index', start_frame_open)
        data_open = dict(
            frame_dir=video_path,
            total_frames=end_frame_open - start_frame_open,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index_open,
            video=vid_open,
            frame_num=frame_num_open,
            filename_tmpl=filename_tmpl,
            modality=modality,
            fmt='video')
        data_open, frame_inds_open = self.pipeline(data_open,fmt='video')

        audio_path = self.base_path + 'rgb/' + self.splits[index] + '/' + self.samples[index][0] + '.wav'
        samples, samplerate = sf.read(audio_path)

        duration = len(samples) / samplerate

        fr_sec = self.samples[index][3].split(':')
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1

        stop_sec = self.samples[index][4].split(':')
        hour1 = float(stop_sec[0])
        minu1 = float(stop_sec[1])
        sec1 = float(stop_sec[2])
        stop_sec = (hour1 * 60 + minu1) * 60 + sec1

        start1 = fr_sec / duration * len(samples)
        end1 = stop_sec / duration * len(samples)
        start1 = int(np.round(start1))
        end1 = int(np.round(end1))
        samples = samples[start1:end1]

        resamples = samples[:160000]
        while len(resamples) < 160000:
            resamples = np.tile(resamples, 10)[:160000]

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram - mean, std + 1e-9)

        audio_path_open = self.base_path_open + self.prefix_list[index_open] + 'audio/' + self.video_list[index_open][:-4] + '.wav'
        start_time = frame_inds_open[0] / 24.0
        end_time = frame_inds_open[-1] / 24.0
        samples, samplerate = sf.read(audio_path_open)
        duration = len(samples) / samplerate
        spectrogram_open = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate)


        label1 = self.samples[index][-1]
        label1_open = int(8)

        return data, spectrogram, label1, data_open, spectrogram_open, label1_open

    def __len__(self):
        # return max(len(self.samples), len(self.samples_open))
        return len(self.samples)

