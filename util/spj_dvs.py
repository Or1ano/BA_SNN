
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import spikingjelly.datasets
import spikingjelly.datasets.cifar10_dvs as cifar10_dvs

root_dir = r'F:\BA_SNN\archs\cifar10_dvs'
dtype = torch.float
cifar10_root = r'F:\BA_SNN\experiment\data\CIFAR10'
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_transform = []
ts_transform = []
train_transform.append(transforms.ToTensor())
# ts_transform.append(transforms.ToTensor())

# 使用origin_set提取cifar10_dvs数据集
origin_set = cifar10_dvs.CIFAR10DVS(root_dir)
# 划分为train和test
# train_set, test_set = spikingjelly.datasets.split_to_train_test_set(train_ratio=0.8, origin_dataset=origin_set, num_classes=10)
#
# print(len(train_set))
# print(len(test_set))

# print(origin_set[0][0])
nplist = []
for i in origin_set[0][0]:
    # 转换为兼容编码顺序
    native_array = origin_set[0][0][i].astype(origin_set[0][0][i].dtype.newbyteorder('='))
    # 转换为float32编码
    converted_array = native_array.astype(np.float32)
    # 将numpy array 转换成 tensor类型
    nplist.append(torch.from_numpy(converted_array))
# 讲整个nplist转换为tensor格式
origin_ts = torch.stack(nplist, dim=0)

# 应用文章提到的transform序列
ts_transform.append(transforms.ToPILImage())
ts_transform.append(transforms.Resize(size=(48, 48)))
ts_transform.append(transforms.ToTensor())
ts_transform = transforms.Compose(ts_transform)
ts_transformed = (ts_transform(origin_ts))



# train_transform.append(normalize)
# train_transform = transforms.Compose(train_transform)
# val_transform = transforms.Compose([transforms.ToTensor(),
#                                         normalize
#                                         ])
# train_data = datasets.CIFAR10(root=cifar10_root, train=True, download=True,
#                                       transform=train_transform)
# val_data = datasets.CIFAR10(root=cifar10_root, train=False, download=True,
#                                     transform=val_transform)
# print(train_data[0][0].unsqueeze(0).to(dtype).shape)
# print(val_data[0][0].unsqueeze(0).to(dtype).shape)

print(origin_ts.shape)
print(ts_transformed.shape)
