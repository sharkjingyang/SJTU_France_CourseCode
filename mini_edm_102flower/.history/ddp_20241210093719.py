import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import argparse

# 设置分布式训练环境
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# 定义一个简单的二次函数回归模型
class SimpleQuadModel(nn.Module):
    def __init__(self):
        super(SimpleQuadModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入为 x，输出为隐藏层
        self.fc2 = nn.Linear(64, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 1)   # 输出为 y

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 生成数据集：y = ax^2 + bx + c
class QuadDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.linspace(-10, 10, num_samples).view(-1, 1)
        self.y = 3 * self.x**2 + 2 * self.x + 1 + torch.randn_like(self.x) * 0.5  # 加入一点噪声

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 训练过程
def train(rank, world_size):
    setup(rank, world_size)

    # 创建数据集和分布式数据加载器
    dataset = QuadDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    # 创建模型
    model = SimpleQuadModel().to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        sampler.set_epoch(epoch)

        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(rank), y_batch.to(rank)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    cleanup()

# 启动训练
if __name__ == "__main__":
    world_size = torch.cuda.device_count()  # 获取 GPU 数量
    torch.multiproce
