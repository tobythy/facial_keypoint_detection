import os
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from random import shuffle

FOLDER_LIST = ['./data/I/', './data/II/']
LABLE = 'label.txt'



def expand_roi(x1, y1, x2, y2, img_width, img_height, ratio = 0.25):
    """
    扩增矩形框
    :param x1: 原矩形框左上角顶点x坐标
    :param y1: 原矩形框左上角顶点y坐标
    :param x2: 原矩形框右下角顶点x坐标
    :param y2: 原矩形框右下角顶点y坐标
    :param img_width: 图像宽度
    :param img_height: 图像高度
    :param ratio: 扩增倍数，默认0.25倍
    """

    width = x2 - x1 + 1
    height = y2 - y1 + 1
    padding_width = int(width * ratio)
    padding_height = int(height * ratio)
    roi_x1 = x1 - padding_width
    roi_y1 = y1 - padding_height
    roi_x2 = x2 + padding_width
    roi_y2 = y2 + padding_height

    # 扩增后的人脸框不要超过图像大小
    roi_x1 = 0 if roi_x1 < 0 else roi_x1
    roi_y1 = 0 if roi_y1 < 0 else roi_y1
    roi_x2 = img_width - 1 if roi_x2 >= img_width else roi_x2
    roi_y2 = img_height - 1 if roi_y2 >= img_height else roi_y2

    return roi_x1, roi_y1, roi_x2, roi_y2, \
           roi_x2 - roi_x1 + 1, roi_y2 - roi_y1 + 1


def remove_invalid_image(lines, folder):
    images = []
    for line in lines:
        name = line.split()[0]
        image_name = os.path.join(folder, name)
        if os.path.isfile(image_name):
            images.append(line)
    return images



def image_bgr_to_rgb(old_img):
    """
    将BGR表示的图像转换成RGB表示的图像
    用于OpenCV与PIL使用的图像格式之间的转换
    """
    (b, g, r) = cv2.split(old_img)
    img_new = cv2.merge((r, g, b))
    return img_new


def expand_images_dict():
    """
    扩增所有图像
    """
    expand_images = []

    for folder in FOLDER_LIST:
        metadata_file = os.path.join(folder, LABLE)
        with open(metadata_file) as f:
            lines = f.readlines()
            lines = remove_invalid_image(lines, folder)
            for line in lines:
                info_list = line.replace("\n", "").split(" ")
                image_name = os.path.join(folder, info_list[0])

                image = cv2.imread(image_name, 1)
                h, w, channel = image.shape

                roi_x1, roi_y1, roi_x2, roi_y2, new_w, new_h = expand_roi(
                    float(info_list[1]), float(info_list[2]), float(info_list[3]), float(info_list[4]), w, h
                )
                image_rect = [roi_x1, roi_y1, roi_x2, roi_y2]
                x = list(map(float, info_list[5::2]))
                y = list(map(float, info_list[6::2]))
                image_landmarks = list(zip(x, y))

                expand_images.append({"name": image_name,
                                      "rect": image_rect,
                                      "landmarks": image_landmarks})
    return expand_images


def data_to_str():
    """
    将图像数据转换为string
    """

    expand_images = expand_images_dict()

    train_test_info = []

    for info in expand_images:

        # 原图位置
        train_test_str = info['name']

        # expand后的人脸边框坐标
        rect = info['rect']
        for rect_coor in rect:
            train_test_str += " " + str(rect_coor)

        # 人脸边框的关键点坐标
        landmarks = info['landmarks']
        for i in range(0, len(landmarks)):
            center = landmarks[i]
            center -= np.array([rect[0], rect[1]])
            for center_coor in center:
                train_test_str += " " + str(center_coor)

        train_test_info.append(train_test_str)

    return train_test_info


def validation(images_info):
    """
    验证正确性
    """
    idx = random.randint(0, len(images_info))
    train_test_val = images_info[idx]
    train_test = train_test_val.split(" ")
    image = cv2.imread(train_test[0], 1)
    print(train_test[0])
    # 画人脸矩形框
    cv2.rectangle(image,
                  (int(float(train_test[1])), int(float(train_test[2]))),
                  (int(float(train_test[3])), int(float(train_test[4]))),
                  (0, 255, 0), thickness=2)
    # 画关键点
    for i in range(0, len(train_test) - 5, 2):
        # 由于关键点坐标是相对于人脸矩形框的，绘制时需要调整
        center = (int(float(train_test[i + 5])) + int(float(train_test[1])),
                  int(float(train_test[i + 1 + 5])) + int(float(train_test[2])))
        cv2.circle(image, center, 2, (0, 0, 255), -1)
    image_new = image_bgr_to_rgb(image)
    plt.imshow(image_new)
    plt.show()


def generate_train_test(train_test_info, train_ratio = 0.8):
    """
    生成训练集和测试集，默认比例8：2
    """
    shuffle(train_test_info)
    split_idx = int(len(train_test_info) * train_ratio)
    with open("train.txt", "w") as f:
        for i in range(split_idx):
            train_info = train_test_info[i]
            f.write(train_info + "\n")
    with open("test.txt", "w") as f:
        for i in range(split_idx, len(train_test_info)):
            test_info = train_test_info[i]
            f.write(test_info + "\n")



if __name__ == '__main__':

    train_test_info = data_to_str()
    validation(train_test_info)
    generate_train_test(train_test_info)

