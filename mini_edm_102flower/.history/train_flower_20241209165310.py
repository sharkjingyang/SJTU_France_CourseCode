import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# 数据集路径
dataset_path = './datasets/102flower/'  # 请替换成你下载并解压数据集的实际路径

# 数据预处理（归一化和转换为Tensor）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),          # 将图片转化为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])

# 加载训练集和测试集
train_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, 'train'),  # 训练集文件夹路径
    transform=transform
)

test_dataset = datasets.ImageFolder(
    os.path.join(dataset_path, 'test'),   # 测试集文件夹路径
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 打印一些信息，检查数据集是否正确加载
print(f"训练集大小: {len(train_loader.dataset)}")
print(f"测试集大小: {len(test_loader.dataset)}")

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
