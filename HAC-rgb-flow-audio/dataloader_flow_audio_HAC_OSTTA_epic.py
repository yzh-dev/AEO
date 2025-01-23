from mmaction.datasets.pipelines import Compose
import torch.utils.data
import csv
import soundfile as sf
from scipy import signal
import numpy as np
import os
import imageio.v3 as iio
import pandas as pd

def load_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().split() for line in lines]
    paths, labels = zip(*data)
    return paths, labels

# source_all = [0, 2, 3, 5, 6]
source_all = [0, 1, 2, 3, 4, 5, 6]
target_all = [0, 1, 2, 3, 4, 5, 6]

target_all_epic = [0, 1, 3, 4, 5, 6, 7]
""" sleeping,0
watching tv,1
eating,2
drinking,3
swimming,4
running,5
opening door,6
 """
 
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
    def __init__(self, split='test', source=True, domain=['human'],  modality='rgb', cfg=None, cfg_flow=None, use_video=True, use_flow=True, use_audio=True, datapath='/path/to/HAC/', datapath_open=''):
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

        for dom in domain:
            prefix = dom + '/'
            with open("splits/HAC_%s_only_%s.csv" % (split, dom)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

            with open("splits/HAC_train_only_%s.csv" % (dom)) as f:
                f_csv = csv.reader(f)
                for i, row in enumerate(f_csv):
                    self.video_list.append(row[0])
                    self.prefix_list.append(prefix)
                    self.label_list.append(row[1])

        # train_file_name_open = "splits/ucf101_open.txt"
        # self.samples_open, self.labels_open = load_txt_file(train_file_name_open)

        data1 = []
        splits = []
        for dom in ['D3']:
            for spl in ['train', 'test']:
                train_file = pd.read_pickle('splits/'+dom+"_"+spl+".pkl")

                for _, line in train_file.iterrows():
                    image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],
                            line['stop_timestamp']]
                    labels = line['verb_class']
                    if int(labels) in target_all_epic:
                        data1.append((image[0], image[1], image[2], image[3], image[4], int(labels)))
                        splits.append(spl)

        # build the data pipeline
        if split == 'train':
            train_pipeline_flow = cfg_flow.data.train.pipeline
            self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            val_pipeline_flow = cfg_flow.data.val.pipeline
            self.pipeline_flow = Compose(val_pipeline_flow)

        self.samples_open = data1
        self.splits_open = splits
        self.cfg = cfg
        self.cfg_flow = cfg_flow
        self.interval = 9
        self.video_path_base = self.base_path + 'HAC/'
        if not os.path.exists(self.video_path_base):
            os.mkdir(self.video_path_base)

    def __getitem__(self, index):
        label1 = int(self.label_list[index])
        video_path = self.video_path_base + self.video_list[index] + "/" 
        video_path = video_path + self.video_list[index] + '-'

        label1_open = int(7)

        video_file_x = self.base_path + self.prefix_list[index] +'flow/' + self.video_list[index][:-4] + '_flow_x.mp4'
        video_file_y = self.base_path + self.prefix_list[index] +'flow/' + self.video_list[index][:-4] + '_flow_y.mp4'
        vid_x = iio.imread(video_file_x, plugin="pyav")
        vid_y = iio.imread(video_file_y, plugin="pyav")

        frame_num = vid_x.shape[0]
        start_frame = 0
        end_frame = frame_num-1

        filename_tmpl_flow = self.cfg_flow.data.val.get('filename_tmpl', '{:06}.jpg')
        modality_flow = self.cfg_flow.data.val.get('modality', 'Flow')
        start_index_flow = self.cfg_flow.data.val.get('start_index', start_frame)
        flow = dict(
            frame_dir=video_path,
            total_frames=end_frame - start_frame,
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index_flow,
            video=vid_x,
            video_y=vid_y,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl_flow,
            modality=modality_flow,fmt='video')
        flow, frame_inds_flow = self.pipeline_flow(flow,fmt='video')

        flow_path_open = self.base_path_open +'flow/'+self.splits_open[index] + '/'+self.samples_open[index][0]
        
        filename_tmpl_flow_open = self.cfg_flow.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        start_index_flow_open = self.cfg_flow.data.train.get('start_index', int(np.ceil(self.samples_open[index][1] / 2)))
        flow_open = dict(
            frame_dir=flow_path_open,
            total_frames=int((self.samples_open[index][2] - self.samples_open[index][1])/2),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index_flow_open,
            filename_tmpl=filename_tmpl_flow_open,
            modality=modality_flow,
            fmt='img')
        flow_open = self.pipeline_flow(flow_open,fmt='img')


        audio_path = self.base_path + self.prefix_list[index] + 'audio/' + self.video_list[index][:-4] + '.wav'
        start_time = frame_inds_flow[0] / 24.0
        end_time = frame_inds_flow[-1] / 24.0
        samples, samplerate = sf.read(audio_path)
        duration = len(samples) / samplerate

        spectrogram = get_spectrogram_piece(samples,start_time,end_time,duration,samplerate)

        audio_path_open = self.base_path_open + 'rgb/' + self.splits_open[index] + '/' + self.samples_open[index][0] + '.wav'

        samples, samplerate = sf.read(audio_path_open)

        duration = len(samples) / samplerate

        fr_sec = self.samples_open[index][3].split(':')
        hour1 = float(fr_sec[0])
        minu1 = float(fr_sec[1])
        sec1 = float(fr_sec[2])
        fr_sec = (hour1 * 60 + minu1) * 60 + sec1

        stop_sec = self.samples_open[index][4].split(':')
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
        frequencies, times, spectrogram_open = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram_open = np.log(spectrogram_open + 1e-7)

        mean = np.mean(spectrogram_open)
        std = np.std(spectrogram_open)
        spectrogram_open = np.divide(spectrogram_open - mean, std + 1e-9)


        return flow, spectrogram, label1, flow_open, spectrogram_open, label1_open

    def __len__(self):
        return len(self.video_list)


