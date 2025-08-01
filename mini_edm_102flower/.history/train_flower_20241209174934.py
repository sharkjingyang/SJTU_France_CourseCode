import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import OxfordFlowers102

# 定义数据预处理流程
transform = transforms.Compose([
    transforms.Resize((224, 224)),       # 将图像调整为统一尺寸
    transforms.ToTensor(),               # 将图像转为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载 Oxford Flowers 102 数据集
dataset = OxfordFlowers102(root='./data', split='train', download=True, transform=transform)

# 创建 DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 示例: 获取一个 batch 的数据
for images, labels in dataloader:
    print(images.shape, labels.shape)  # images: (batch_size, 3, 224, 224), labels: (batch_size,)
    break
