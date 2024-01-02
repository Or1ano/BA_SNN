import os
import numpy as np
import torch

# 定义路径
base_path = r'F:\BA_SNN\archs\cifar10_dvs\events_np'
train_path = r'F:\BA_SNN\experiment\data\CIFAR10DVS\events_pt\train'
test_path = r'F:\BA_SNN\experiment\data\CIFAR10DVS\events_pt\test'

# 确保输出目录存在
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)
print("输出目录已创建或已存在。")

# 遍历每个子文件夹
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    target = str(folder)
    print(f"正在处理文件夹: {folder}")

    # 读取所有npz文件并排序
    npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    print(f"找到 {len(npz_files)} 个npz文件。")
    # 分别处理训练和测试文件
    for dataset_type, file_range, output_path in [('train', range(800), train_path),
                                                  ('test', range(800, 1000), test_path)]:
        data_list = []

        for file_index in file_range:
            file_path = os.path.join(folder_path, npz_files[file_index])
            with np.load(file_path) as data:
                # 需要从npz文件中提取所有数组
                nplist = []
                for i in data:
                    # 转换为兼容编码顺序
                    native_array = data[i].astype(data[i].dtype.newbyteorder('='))
                    # 转换为float32编码
                    converted_array = native_array.astype(np.float32)
                    # 将numpy array 转换成 tensor类型
                    nplist.append(torch.from_numpy(converted_array))
                # 将整个nplist转换为tensor格式
                origin_ts = torch.stack(nplist, dim=0)
                # print(origin_ts.shape)
                # 将提取出来的origin_ts赋值给 data序列
                data_list.append(origin_ts)
                # print(f"文件 {npz_files[file_index]} 已处理。")

        # 合并为一个Tensor
        # 找出第二维上的最小值
        min_size = min(tensor.size(1) for tensor in data_list)
        # 裁剪所有Tensor到最小
        cropped_data_list = [tensor[:, :min_size] for tensor in data_list]
        # 合并
        combined_tensor = torch.stack(cropped_data_list, dim=0)
        print(combined_tensor.shape)
        # 保存Tensor
        output_filename = os.path.join(output_path, f'{folder}_{dataset_type}.pt')
        torch.save({'data': combined_tensor, 'targets': target}, os.path.join(output_path, f'{folder}_{dataset_type}.pt'))
        print(f"{dataset_type} 数据已保存到 {output_filename}")
