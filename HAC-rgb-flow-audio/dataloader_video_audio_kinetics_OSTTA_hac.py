from mmaction.datasets.pipelines import Compose
import torch.utils.data
import csv
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio
import pandas as pd
from pydub import AudioSegment
import io
import random

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split() for line in lines]
    paths, labels = zip(*data)
    return paths, labels

def load_txt_file_kinetics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().rsplit(' ', 1) for line in lines]
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



class HACDOMAIN(torch.utils.data.Dataset):
    def __init__(self, split='test', cfg=None, cfg_flow=None, use_kinetics_100=False, use_random_noise=False, use_video=True, use_flow=True, use_audio=True, video_noise_type='gaussian_noise', audio_noise_type='gaussian_noise', datapath='/path/to/HAC/', datapath_open=''):
        self.base_path = datapath
        self.base_path_open = datapath_open
        self.video_list = []
        self.prefix_list = []
        self.label_list = []
        self.video_list_open = []
        self.prefix_list_open = []
        self.label_list_open = []
        self.use_video = use_video
        self.use_audio = use_audio
        self.use_flow = use_flow
        self.audio_noise_type = audio_noise_type
        self.video_noise_type = video_noise_type
        self.use_kinetics_100 = use_kinetics_100

        self.video_noise_types = ["defocus_blur", "frost", "brightness", "pixelate", "jpeg_compression", "gaussian_noise_5"]
        self.audio_noise_types = ["wind", "traffic", "thunder", "rain", "crowd", "gaussian_noise"]
        self.use_random_noise = use_random_noise

        if self.use_kinetics_100:
            train_file_name = "splits/Kinetics_test_100.txt"
        else:
            train_file_name = "splits/Kinetics_val.txt"

        self.samples, self.labels = load_txt_file_kinetics(train_file_name)

        for dom in ['animal', 'cartoon']:
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
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

            with open(self.base_path_open + "HAC_Splits/HAC_train_only_%s.csv" % (dom)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

        # build the data pipeline
        if split == 'train':
            train_pipeline = cfg.data.train.pipeline
            self.pipeline = Compose(train_pipeline)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.pipeline = Compose(val_pipeline)

        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.interval = 9
        self.video_path_base = self.base_path + 'HAC/'
        if not os.path.exists(self.video_path_base):
            os.mkdir(self.video_path_base)

    def __getitem__(self, index):
        label1 = int(self.labels[index])

        if self.use_kinetics_100:
            label1_open = int(100)
        else:
            label1_open = int(50)

        index_open = index
        if index >= len(self.video_list):
            index_open = index - len(self.video_list)

        video_path = ''

        if self.video_noise_type == 'None':
            video_file = self.base_path + 'Kinetics-600-train/' + self.samples[index]
        else:
            #video_file = '/cluster/scratch/hadong/OSTTA/Kinetics50-C/video-C/' + self.video_noise_type + '/' + self.samples[index]
            video_file = '/cluster/work/ibk_chatzi/hao/dataset/OSTTA/Kinetics50-C/video-C/' + self.video_noise_type + '/' + self.samples[index]
        vid = iio.imread(video_file, plugin="pyav")

        frame_num = vid.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            label=-1,
            start_index=start_index,
            video=vid,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality,fmt='video')
        data, frame_inds = self.pipeline(data,fmt='video')

        if self.video_noise_type == 'None':
            video_file_open = self.base_path_open + self.prefix_list[index_open] +'videos/' + self.video_list[index_open]
        else:
            #video_file_open = '/cluster/scratch/hadong/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'video-C/' + self.video_noise_type + '/' + self.video_list[index_open]
            if self.use_random_noise:
                random_noise = random.choice(self.video_noise_types)
                video_file_open = '/cluster/work/ibk_chatzi/hao/dataset/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'video-C/' + random_noise + '/' + self.video_list[index_open]
            else:
                video_file_open = '/cluster/work/ibk_chatzi/hao/dataset/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'video-C/' + self.video_noise_type + '/' + self.video_list[index_open]
        vid_open = iio.imread(video_file_open, plugin="pyav")

        frame_num = vid_open.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl = self.cfg.data.val.get('filename_tmpl', '{:06}.jpg')
        modality = self.cfg.data.val.get('modality', 'RGB')
        start_index = self.cfg.data.val.get('start_index', start_frame)
        data_open = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            video=vid_open,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality,fmt='video')
        data_open, frame_inds_open = self.pipeline(data_open,fmt='video')


        audio_path = self.base_path + 'Kinetics-600-train/' + self.samples[index][:-4] + '.wav'
        start_time = frame_inds[0] / 24.0
        end_time = frame_inds[-1] / 24.0

        if self.audio_noise_type == 'None':
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate
        elif self.audio_noise_type == 'gaussian_noise':
            samples, samplerate = sf.read(audio_path)
            duration = len(samples) / samplerate
            noise_std = 0.38
            # noise_std = 0.18
            # noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]
            # noise_std = 0.08
            noise = np.random.normal(0, noise_std, len(samples))
            samples = samples + noise
        else:
            # audio_path = '/cluster/scratch/hadong/OSTTA/Kinetics50-C/audio-C/' + self.audio_noise_type + '/' + self.samples[index][:-4] + '.wav'
            audio_path = '/cluster/work/ibk_chatzi/hao/dataset/OSTTA/Kinetics50-C/audio-C/' + self.audio_noise_type + '/' + self.samples[index][:-4] + '.wav'
            samples, samplerate = sf.read(audio_path)
            samples = samples[:, 0]
            duration = len(samples) / samplerate

        spectrogram = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate)
        
        
        audio_path_open = self.base_path_open + self.prefix_list[index_open] + 'audio/' + self.video_list[index_open][:-4] + '.wav'
        start_time = frame_inds_open[0] / 24.0
        end_time = frame_inds_open[-1] / 24.0

        if self.use_random_noise:
            random_noise = random.choice(self.audio_noise_types)
            if random_noise == 'gaussian_noise':
                samples, samplerate = sf.read(audio_path_open)
                duration = len(samples) / samplerate
                noise_std = 0.38
                # noise_std = 0.18
                # noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]
                # noise_std = 0.08
                noise = np.random.normal(0, noise_std, len(samples))
                samples = samples + noise
            else:
                # audio_path_open = '/cluster/scratch/hadong/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'audio-C/' + self.audio_noise_type + '/' + self.video_list[index_open][:-4] + '.wav'
                audio_path_open = '/cluster/work/ibk_chatzi/hao/dataset/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'audio-C/' + random_noise + '/' + self.video_list[index_open][:-4] + '.wav'
                samples, samplerate = sf.read(audio_path_open)
                samples = samples[:, 0]
                duration = len(samples) / samplerate
        else:
            if self.audio_noise_type == 'None':
                samples, samplerate = sf.read(audio_path_open)
                duration = len(samples) / samplerate
            elif self.audio_noise_type == 'gaussian_noise':
                samples, samplerate = sf.read(audio_path_open)
                duration = len(samples) / samplerate
                noise_std = 0.38
                # noise_std = 0.18
                # noise_std = [.08, .12, 0.18, 0.26, 0.38][intensity - 1]
                # noise_std = 0.08
                noise = np.random.normal(0, noise_std, len(samples))
                samples = samples + noise
            else:
                # audio_path_open = '/cluster/scratch/hadong/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'audio-C/' + self.audio_noise_type + '/' + self.video_list[index_open][:-4] + '.wav'
                audio_path_open = '/cluster/work/ibk_chatzi/hao/dataset/OSTTA/HAC-C/' + self.prefix_list[index_open] + 'audio-C/' + self.audio_noise_type + '/' + self.video_list[index_open][:-4] + '.wav'
                samples, samplerate = sf.read(audio_path_open)
                samples = samples[:, 0]
                duration = len(samples) / samplerate

        spectrogram_open = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate)

        return data, spectrogram, label1, data_open, spectrogram_open, label1_open

    def __len__(self):
        return len(self.samples)


