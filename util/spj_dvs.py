
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import spikingjelly.datasets
import spikingjelly.datasets.cifar10_dvs as cifar10_dvs

npz_path = r'F:\BA_SNN\archs\cifar10_dvs\frames_number_10_split_by_number\airplane\cifar10_airplane_0.npz'
pt_path = r'F:\BA_SNN\experiment\data\CIFAR10DVS\events_pt\train\0.pt'
root_dir = r'F:\BA_SNN\archs\cifar10_dvs'
dtype = torch.float
cifar10_root = r'F:\BA_SNN\experiment\data\CIFAR10'
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transform = []
ts_transform = []
frame_ts_transformed_list = []
train_transform.append(transforms.ToTensor())
# ts_transform.append(transforms.ToTensor())

# 使用origin_set提取cifar10_dvs数据集
origin_set = cifar10_dvs.CIFAR10DVS(root_dir)
frame_set = cifar10_dvs.CIFAR10DVS(root=root_dir, data_type='frame', frames_number=10, split_by='number')

print(len(origin_set))
print(origin_set[0][0])

print(len(frame_set))
print(frame_set[0][0].shape)
# print(frame_set[0][0])

frame_npnd = frame_set[0][0]
frame_ts = torch.from_numpy(frame_npnd)
print('转换前： ')
print(frame_ts.shape)

# 应用data提到的transform序列
ts_transform.append(transforms.ToPILImage())
ts_transform.append(transforms.Resize(size=(48, 48)))
ts_transform.append(transforms.ToTensor())
ts_transform = transforms.Compose(ts_transform)

for t in range(frame_ts.size(0)):
    # print(t)
    frame_ts_transformed_list.append(ts_transform(frame_ts[t, ...]))
print('转换后： ')
print(frame_ts_transformed_list[0].shape)
frame_ts_transformed = torch.stack(frame_ts_transformed_list, dim=0)
print(frame_ts_transformed.shape)

data = np.load(npz_path)
print(f"{npz_path}中含有的数据如下：")
for i in data['frames']:
    print(i.shape)

frame_ts = torch.load(pt_path)
print(frame_ts)
f_data = frame_ts['data']
f_target = torch.tensor(frame_ts['target'])
print(f_data.shape)
print(f_target)
