import numpy as np
import cv2

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import random

# folder_list = ['I', 'II']
# folder_list = {'train':'./data/I/', 'test':'./data/II/'}
train_boarder = 112


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
            image.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


class FaceLandmarksDataset(Dataset):
    # Face Landmarks Dataset
    def __init__(self, src_lines, phase, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_name, rect, landmarks = parse_line(self.lines[idx])
        # image
        img = Image.open(img_name).convert('L')
        # img = Image.open(folder_list[self.phase] + img_name).convert('L')
        # img = cv2.imread(img_name)
        # img_crop = img[rect[1]:rect[3], rect[0]:rect[2], :]
        img_crop = img.crop(tuple(rect))
        landmarks = np.array(landmarks).astype(np.float32)

        # you should let your landmarks fit to the train_boarder(112)
        # please complete your code under this blank
        # your code:
        origin_width, origin_height = img_crop.size
        width_ratio = train_boarder / origin_width
        height_ratio = train_boarder / origin_height

        for i in range(0, len(landmarks), 2):
            # 将坐标与crop之后的image对齐
            # landmarks[i] -= rect[0]
            # landmarks[i+1] -= rect[1]

            # 将坐标缩放到resize之后的尺寸
            landmarks[i] = round(landmarks[i] * width_ratio)
            landmarks[i + 1] = round(landmarks[i + 1] * height_ratio)

        # landmarks = landmarks.reshape(-1, 2)  # 转成x,y格式，便于后面操作
        # landmarks[:, 0] -= rect[0]  # 将x坐标与crop之后的image对齐
        # landmarks[:, 1] -= rect[1]  # 将y坐标与crop之后的image对齐
        # ori_h, ori_w, _ = img_crop.shape
        # landmarks[:, 0] *= 1.0 * train_boarder / ori_w  # 将x坐标缩放到resize之后的尺寸
        # landmarks[:, 1] *= 1.0 * train_boarder / ori_h  # 将y坐标缩放到resize之后的尺寸
        #
        #

        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)
        return sample


def load_data(phase):
    data_file = phase + '.txt'
    # data_file = folder_list[phase] + 'label.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            Normalize(),  # do channel normalization
            ToTensor()]  # convert to torch type: NxCxHxW
        )
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, phase, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train')
    # for i in range(1, len(train_set)):
    #     sample = train_set[i]
    #     img = sample['image'].squeeze(0).numpy().transpose((1, 2, 0))
    #     landmarks = sample['landmarks']

    # 随机选取一张图片做测试
    idx_test = random.randint(0, len(train_set))
    sample = train_set[idx_test]
    # img = sample['image'].squeeze(0).numpy().transpose((1, 2, 0))
    img = sample['image']
    landmarks = sample['landmarks']
    # 请画出人脸crop以及对应的landmarks
    # please complete your code under this blank

    img = img.numpy().transpose((1, 2, 0))
    # cv2.circle输出内存是连续的，调用copy使输入为连续内存，输出输入一致
    img_test = img.copy()

    for i in range(0, len(landmarks), 2):
        center = (int(landmarks[i]), int(landmarks[i + 1]))
        cv2.circle(img_test, center, 1, (255, 0, 0), -1)
    cv2.imshow("face", img_test)

        # landmarks = landmarks.reshape(-1, 2)
        # for x, y in landmarks:
        #     center = int(x), int(y)
        #     cv2.circle(img, center, 1, (0, 255, 0), -1)
        # cv2.imshow("face", img)
        #
    key = cv2.waitKey()
    if key == 27:
        exit(0)
    cv2.destroyAllWindows()