import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader

# 定义ToyModel，适用于回归
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(1, 128)  # 输入1个特征（x）
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出1个预测值（y）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # 不使用激活函数进行回归
        return x

# 生成二次函数数据集
class QuadraticDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.linspace(-10, 10, num_samples).view(-1, 1)  # 生成[-10, 10]范围的x
        self.y = self.x**2 + 3*self.x + 5  # 真实的二次函数y = x^2 + 3x + 5
        self.y += 0.1 * torch.randn_like(self.y)  # 添加一些噪声
     
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 获取数据集
def get_dataset():
    trainset = QuadraticDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=32, num_workers=2, sampler=train_sampler)
    return trainloader

# 初始化DDP
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

# 初始化分布式进程组
dist.init_process_group(
    backend='nccl',
    rank=args.local_rank,
    world_size=torch.cuda.device_count()
)

# 加载数据集
trainloader = get_dataset()

# 构建模型
model = ToyModel().to('cuda')
model = DDP(model, device_ids=[args.local_rank])

# 定义优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.MSELoss().to('cuda')  # 使用均方误差损失

# 训练模型
model.train()
iterator = tqdm(range(100))  # 迭代 100 轮
for epoch in iterator:
    trainloader.sampler.set_epoch(epoch)
    for data, label in trainloader:
        data, label = data.to('cuda'), label.to('cuda')
        
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()
        
        if dist.get_rank() == 0:
            iterator.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")
