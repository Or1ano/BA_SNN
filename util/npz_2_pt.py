import os
import numpy as np
import torch

# 定义路径
base_path = r'F:\BA_SNN\archs\cifar10_dvs\frames_number_10_split_by_number'
train_path = r'F:\BA_SNN\experiment\data\CIFAR10DVS\events_pt\train'
test_path = r'F:\BA_SNN\experiment\data\CIFAR10DVS\events_pt\test'

# 确保输出目录存在
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
print("输出目录已创建或已存在。")

target = 0
train_index = 0
test_index = 0

# 遍历每个子文件夹
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    print(f"正在处理文件夹: {folder}")

    # 读取所有npz文件并排序
    npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    print(f"找到 {len(npz_files)} 个npz文件。")
    # 分别处理训练和测试文件
    for dataset_type, file_range, output_path in [('train', range(800), train_path),
                                                  ('test', range(800, 1000), test_path)]:
        # data_list = []

        for file_index in file_range:
            file_path = os.path.join(folder_path, npz_files[file_index])
            with np.load(file_path) as data:
                # 需要从npz文件中提取所有数组
                nplist = []
                for i in data['frames']:
                    # 将numpy array 转换成 tensor类型
                    nplist.append(torch.from_numpy(i))
                # 将整个nplist转换为tensor格式
                frames_ts = torch.stack(nplist, dim=0)
                # print(frames_ts.shape)
                print(f"文件 {npz_files[file_index]} 已处理。")
                # 保存Tensor
                if dataset_type == 'train':
                    output_filename = os.path.join(output_path, f'{train_index}.pt')
                    torch.save({'data':  frames_ts, 'target': target},
                           os.path.join(output_path, f'{train_index}.pt'))
                    print(f"{dataset_type}_{target} 数据已保存到 {output_filename}")
                    train_index += 1
                elif dataset_type == 'test':
                    output_filename = os.path.join(output_path, f'{test_index}.pt')
                    torch.save({'data': frames_ts, 'target': target},
                               os.path.join(output_path, f'{test_index}.pt'))
                    print(f"{dataset_type}_{target} 数据已保存到 {output_filename}")
                    test_index += 1

        # # 合并为一个Tensor
        # # 找出第二维上的最小值
        # min_size = min(tensor.size(1) for tensor in data_list)
        # # 裁剪所有Tensor到最小
        # cropped_data_list = [tensor[:, :min_size] for tensor in data_list]
        # # 合并
        # combined_tensor = torch.stack(cropped_data_list, dim=0)
        # print(combined_tensor.shape)

    target += 1
