import os
import re
import random
import cv2
import torch
from torch.utils import data
from torch import nn

# For creating a custom dataset: it needs to contain three funcs: __init__, __len__, __getitem__
# Default: no scale ratio
class Tactile_Vision_dataset(data.Dataset):
    def __init__(self, scale_ratio = 1, video_length = 8, data_path='./data'):
        self.data_path = data_path
        self.label_files = []
        self.train_data = []
        self.scale_percent = scale_ratio
        self.video_length = video_length
        for root, dirs, files in os.walk(data_path, topdown=True):
            for file in files:
                if file.endswith('.dat'):
                    self.label_files.append(os.path.join(root, file))
        self.label_files.sort()
        pat = re.compile(r'object([0-9]+)_result')  #filter
        for label_file in self.label_files:
            idx = pat.search(label_file).group(1)
            fp = open(label_file, 'r')
            lines = fp.readlines()
            self.train_data.extend([line.replace('\n','') + ' ' + idx for line in lines])

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):  #need to be defined to let data_loader work
        train_data = self.train_data[index]
        output_tactile_imgs = []
        output_rgb_imgs = []
        train_data = train_data.split(' ')
        object_id = train_data[-1]  # object ID
        id_2 = train_data[-3] # Experiment ID (on the same object)
        status = int(train_data[2][0]) # Label  # 0 -> slip; 1 -> success
        label = torch.tensor([status]).long()
        label = torch.squeeze(label)
        path = os.path.join(self.data_path, 'object' + object_id, id_2 + '_mm')
        rgb_img_paths = []
        for root, dirs, files in os.walk(path, topdown=True):
            for file in files:
                if ("external_" in file) and file.endswith('.jpg'):  # select camera images
                    rgb_img_paths.append(os.path.join(root, file))
        rgb_img_paths_selected = []
        rgb_img_paths.sort()  #Sort images
        start_image = train_data[5]
        index = 0
        while(len(rgb_img_paths_selected) < self.video_length):  # 8 frames per time (LSTM)
            if start_image in rgb_img_paths[index] or len(rgb_img_paths_selected) > 0:
                rgb_img_paths_selected.append(rgb_img_paths[index])
            index += 1
        index = 0
        for rgb_img_path in rgb_img_paths_selected:
            cor_tactile_img_path = rgb_img_path.replace('external', 'gelsight')
            rgb_img = cv2.imread(rgb_img_path)
            tactile_img = cv2.imread(cor_tactile_img_path)
            size = rgb_img.shape  # 480, 640, 3 (width, height, channel)
            # new width / new height = 480 / 640 * scale_percent

            rgb_img_resized = cv2.resize(rgb_img, (int(size[1] * self.scale_percent), int(size[0] * self.scale_percent)), interpolation = cv2.INTER_AREA)
            # rgb_img_resized = cv2.resize(rgb_img,(224, 224),interpolation=cv2.INTER_AREA)
            tactile_img_resized = cv2.resize(tactile_img, (int(size[1] * self.scale_percent), int(size[0] * self.scale_percent)), interpolation=cv2.INTER_AREA)
            size = rgb_img_resized.shape
            # size = tactile_img_resized.shape
            rgb_img_tensor = torch.from_numpy(rgb_img_resized.transpose(2,0,1)).float()

            #turn into a tensor (3, 240, 320)  -> resized one
            tactile_img_tensor = torch.from_numpy(tactile_img_resized.transpose(2,0,1)).float()
            if index == 0:
                output_rgb_imgs = rgb_img_tensor[None,:]
                output_tactile_imgs = tactile_img_tensor[None,:]
            else:
                output_rgb_imgs = torch.cat([output_rgb_imgs, rgb_img_tensor[None,:]], dim=0)
                output_tactile_imgs = torch.cat([output_tactile_imgs, tactile_img_tensor[None,:]], dim=0)
            index += 1
        return output_rgb_imgs.transpose(0, 1), output_tactile_imgs.transpose(0, 1), label # rgb images; visual images; label

if __name__ == "__main__":
    # set a global dataset path
    train_dataset = Tactile_Vision_dataset(data_path = '/home/yhan389/Desktop/SlipDetection/Small Dataset/training')
    print(train_dataset[0][0])
    # for i in range(2):
    #     output_rgb_imgs, output_tactile_imgs, label = train_dataset[i]
    #     print(output_rgb_imgs[0].shape)
    #     print(output_tactile_imgs[0].shape)

    # for i in range(1000):
    #     train_dataset[i]
    #     print(i)
