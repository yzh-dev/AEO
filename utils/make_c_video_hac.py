import numpy as np
import os
import argparse
import cv2
import collections

def gaussian_noise(image, severity=5):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    image = np.array(image) / 255.
    image = np.clip(image + np.random.normal(size=image.shape, scale=c), 0, 1) * 255
    return image.astype(np.uint8)


def pixelate(image, severity=5):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    height, width, _ = image.shape
    small = cv2.resize(image, (int(width * c), int(height * c)), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    return pixelated


def brightness(image, severity=5):
    c = [1.1, 1.2, 1.3, 1.4, 1.5][severity - 1]

    image_float = image.astype(np.float32)
    image_bright = image_float * c
    image_bright = np.clip(image_bright, 0, 255).astype(np.uint8)
    return image_bright


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

def defocus_blur(x, severity=5):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    image = np.clip(channels, 0, 1) * 255
    return image.astype(np.uint8)


def frost(frame, idx=0, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)

    frost = cv2.resize(frost, (frame.shape[1], frame.shape[0]))
    mixed_frame = np.clip(c[0] * np.array(frame) + c[1] * frost, 0, 255)

    return mixed_frame.astype(np.uint8)


def jpeg_compression(frame, severity=5):
    c = [25, 18, 15, 10, 7][severity - 1]

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), c]
    _, encoded_img = cv2.imencode('.jpg', frame, encode_param)
    
    # Decode the JPEG image back to a NumPy array
    compressed_frame = cv2.imdecode(encoded_img, 1)
    
    return compressed_frame

def load_txt_file_kinetics(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = [line.strip().rsplit(' ', 1) for line in lines]
    paths, labels = zip(*data)
    return paths, labels

train_file_name = "HAC-rgb-flow-audio/splits/Kinetics_val.txt"
samples, labels = load_txt_file_kinetics(train_file_name)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--corruption', type=str, default='gaussian_noise', help='Type of corruption to apply')
parser.add_argument('--severity', type=int, default=3, choices=[1, 2, 3, 4, 5], help='Severity of corruption to apply')
parser.add_argument('--data_path', type=str, default='HAC/cartoon/videos', help='Path to test data')
parser.add_argument('--save_path', type=str, default='HAC-C/cartoon/video-C', help='Path to store corruption data')
args = parser.parse_args()


if args.corruption == 'gaussian_noise':
    corruption_method = gaussian_noise
elif args.corruption == 'defocus_blur':
    corruption_method = defocus_blur
elif args.corruption == 'frost':
    corruption_method = frost
elif args.corruption == 'brightness':
    corruption_method = brightness
elif args.corruption == 'pixelate':
    corruption_method = pixelate
elif args.corruption == 'jpeg_compression':
    corruption_method = jpeg_compression

dir = args.data_path
samples = os.listdir(dir)

save_path = os.path.join(args.save_path, args.corruption)
#save_path = os.path.join(args.save_path, 'gaussian_noise_5')

if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except:
        print(save_path)

for i in range(len(samples)):
    input_video_path = args.data_path + '/' + samples[i]
    output_video_path = save_path + '/' + samples[i]
    
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for the output video

    # Create a VideoWriter object to write the video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    idx = np.random.randint(5)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.corruption == 'frost':
            noisy_frame = corruption_method(frame, idx, severity=args.severity)
        else:
            noisy_frame = corruption_method(frame, severity=args.severity)

        out.write(noisy_frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    print(f"Noisy video saved as {output_video_path}")
