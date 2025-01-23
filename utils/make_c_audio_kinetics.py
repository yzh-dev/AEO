import numpy as np
from pydub import AudioSegment
import soundfile as sf
import os
import torch.utils.data as data
import torch
import argparse
import collections

def load_txt_file_kinetics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().rsplit(' ', 1) for line in lines]
    paths, labels = zip(*data)
    return paths, labels

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--corruption', type=str, default='gaussian_noise', choices=['all', 'gaussian_noise', 'traffic', 'crowd', 'rain', 'thunder', 'wind'], help='Type of corruption to apply')
parser.add_argument('--data_path', type=str, default='data_path/VGGSound/image_mulframe_test', help='Path to test data')
parser.add_argument('--save_path', type=str, default='data_path/VGGSound/image_mulframe_test-C', help='Path to store corruption data')
parser.add_argument('--severity', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Severity of corruption to apply')
parser.add_argument('--weather_path', type=str, default='NoisyAudios/', help='Path to store corruption data')
args = parser.parse_args()

train_file_name = "./HAC-rgb-flow-audio/splits/Kinetics_test_100.txt"
samples, labels = load_txt_file_kinetics(train_file_name)

print(len(samples))
print(samples[0][:-4] + '.wav')


def add_external_noise(audio_path, weather_path, output_path, intensity):
    audio = AudioSegment.from_file(audio_path)
    rain_sound = AudioSegment.from_file(weather_path)

    # adjust the length
    if len(audio) <= len(rain_sound):
        rain_sound = rain_sound[:len(audio)]
    else:
        print(len(audio), len(rain_sound))
        num_repeats = len(audio) // len(rain_sound) + 1
        rain_sound = rain_sound * num_repeats
        rain_sound = rain_sound[:len(audio)]
        print(len(audio), len(rain_sound))

    #rain_sound = rain_sound.apply_gain(-8)
    scale = [-16, -14, -12, -10, -8]
    rain_sound = rain_sound.apply_gain(scale[intensity-1])

    output = audio.overlay(rain_sound)
    output = output.set_frame_rate(16000)

    output.export(output_path, format="wav")

#save_path = '/cluster/scratch/hadong/OSTTA/Kinetics50-C/audio-C/'
save_path = args.save_path
save_path = os.path.join(save_path, args.corruption)
if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except:
        print(save_path)
for i in range(len(samples)):
    #audio_path = '/cluster/work/ibk_chatzi/hao/dataset/video_datasets/Kinetics-600-train/' + samples[i][:-4] + '.wav'
    audio_path = args.data_path + samples[i][:-4] + '.wav'
    output_path = save_path + '/' + samples[i][:-4] + '.wav'
    output_r_path = samples[i][:-4].split('/')[0]
    save_path_r = os.path.join(save_path, output_r_path)
    if not os.path.exists(save_path_r):
        try:
            os.makedirs(save_path_r)
        except:
            print(save_path_r)
    add_external_noise(audio_path, os.path.join(args.weather_path, args.corruption + '.wav'), output_path, args.severity)
