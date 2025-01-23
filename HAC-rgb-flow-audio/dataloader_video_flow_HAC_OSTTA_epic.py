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
            train_pipeline = cfg.data.train.pipeline
            self.pipeline = Compose(train_pipeline)
            train_pipeline_flow = cfg_flow.data.train.pipeline
            self.pipeline_flow = Compose(train_pipeline_flow)
        else:
            val_pipeline = cfg.data.val.pipeline
            self.pipeline = Compose(val_pipeline)
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

        video_file = self.base_path + self.prefix_list[index] +'videos/' + self.video_list[index]
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
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index,
            video=vid,
            frame_num=frame_num,
            filename_tmpl=filename_tmpl,
            modality=modality,fmt='video')
        data, frame_inds = self.pipeline(data,fmt='video')

        video_path_open = self.base_path_open +'rgb/'+self.splits_open[index] + '/'+self.samples_open[index][0]
        flow_path_open = self.base_path_open +'flow/'+self.splits_open[index] + '/'+self.samples_open[index][0]
        # video_file_open = self.base_path_open + 'video/' + self.samples_open[index]
        # vid_open = iio.imread(video_file_open, plugin="pyav")

        filename_tmpl_open = self.cfg.data.train.get('filename_tmpl', 'frame_{:010}.jpg')
        start_index_open = self.cfg.data.train.get('start_index', int(self.samples_open[index][1]))
        data_open = dict(
            frame_dir=video_path_open,
            total_frames=int(self.samples_open[index][2] - self.samples_open[index][1]),
            # assuming files in ``video_path`` are all named with ``filename_tmpl``  # noqa: E501
            label=-1,
            start_index=start_index_open,
            filename_tmpl=filename_tmpl_open,
            modality=modality,
            fmt='img')
        data_open = self.pipeline(data_open,fmt='img')


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

        return data, flow, label1, data_open, flow_open, label1_open

    def __len__(self):
        return len(self.video_list)


