from mmaction.datasets.pipelines import Compose
import torch.utils.data
import pandas as pd
import soundfile as sf
from scipy import signal
import numpy as np

dom="D3"
spl="test"
target_all=[0, 1, 2, 4, 5, 6, 7]


train_file = pd.read_pickle('./splits/'+dom+"_"+spl+".pkl")#读取当前的划分数据
# 帮我创建一个csv文件，逐行保存data1中的数据
data1=[]
with open('./splits/'+dom+"_"+spl+".csv", 'w') as f:
    for _, line in train_file.iterrows():
        image = [dom + '/' + line['video_id'], line['start_frame'], line['stop_frame'], line['start_timestamp'],line['stop_timestamp']]
        labels = line['verb_class']
        if int(labels) in target_all:
            f.write(f"{image[0]},{image[1]},{image[2]},{image[3]},{image[4]},{int(labels)}\n")
print("CSV文件已保存") 
