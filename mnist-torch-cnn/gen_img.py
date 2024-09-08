import numpy as np
import argparse
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader

import cv2
from PIL import Image


class LocalDataset(Dataset):
    def __init__(self, root_dir, record_dir, transform):
        super(LocalDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.datas = self.read_samples_from_record(root_dir, record_dir)

    def __getitem__(self, index):
        path, target = self.datas[index]
        sample = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.datas)

    def read_samples_from_record(self, root_dir, record_dir):
        samples = []
        with open(record_dir, "r") as f:
            for index, line in enumerate(f):
                line = line.split()
                if len(line) < 2:
                    print("Error, Label is missing")
                    exit()
                else:
                    image_dir, label = line[0], line[1]
                label = int(label)
                image_dir = os.path.join(root_dir, image_dir)
                samples.append((image_dir, label))
        return samples


# 取出一个随机的图片，保存
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)

# 创建保存目录
if not os.path.exists("output_images"):
    os.makedirs("output_images")

# 取出一个batch的数据
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 从batch中随机选取一张图片
random_index = torch.randint(0, images.size(0), (1,)).item()
random_image = images[random_index]

# 保存图片
output_path = os.path.join("output_images", "random_image.png")
cv_image = random_image.squeeze().numpy()
cv_image = cv_image * 0.5 + 0.5
cv_image = cv_image * 255
cv_image = cv_image.astype(np.uint8)
cv2.imwrite(output_path, cv_image)

print(f"Random image saved to {output_path}")

image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
assert image.shape == (28, 28), f"Image shape is {image.shape}, expected (28, 28)"
print("Image is of correct shape (28, 28)")
