import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# 数据集路径
dataset_path = './datasets/102flower/'  # 请替换成你下载并解压数据集的实际路径
image_folder= './datasets/102flower/'
train_path = os.path.join(dataset_path, 'train')
test_path = os.path.join(dataset_path, 'test')

# 创建 train 和 test 文件夹
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 按比例划分数据集（例如 80% 训练集，20% 测试集）
import random
from shutil import copyfile

# 获取所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

# 随机划分为训练集和测试集
for image_file in image_files:
    if random.random() < 0.8:  # 80% 用作训练
        copyfile(os.path.join(image_folder, image_file), os.path.join(train_path, image_file))
    else:  # 20% 用作测试
        copyfile(os.path.join(image_folder, image_file), os.path.join(test_path, image_file))

print("数据集已划分为训练集和测试集。")


# 数据预处理（归一化和转换为Tensor）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),          # 将图片转化为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载训练集和测试集
train_dataset = datasets.ImageFolder(
    dataset_path,
    transform=transform
)


# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 打印一些信息，检查数据集是否正确加载
print(f"训练集大小: {len(train_loader.dataset)}")

# 可视化部分图像
import matplotlib.pyplot as plt
def show_batch(data_loader):
    images, labels = next(iter(data_loader))
    fig, axes = plt.subplots(1, 5, figsize=(12, 6))
    for i in range(5):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0))  # 重新排列维度为(H, W, C)进行显示
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.savefig("./diagnose/data_flower.png")

# 显示训练集中一批图像
show_batch(train_loader)
