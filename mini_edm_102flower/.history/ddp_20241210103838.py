import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# 配置分布式环境
def init_process(rank, size, fn, backend='nccl'):
    print("22222")

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12567'

    dist.init_process_group(backend, rank=rank, world_size=size)
    print("33333")

    fn(rank, size)

# 简单的二次回归数据集
class QuadraticDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.x = torch.linspace(-10, 10, n_samples).unsqueeze(1)  # 输入: -10到10
        self.y = self.x ** 2  # 标签: 二次函数

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 定义简单的二次函数回归模型
class QuadraticModel(nn.Module):
    def __init__(self):
        super(QuadraticModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练函数
def train(rank, size):
    # 设置GPU
    torch.cuda.set_device(rank)
    model = QuadraticModel().to(rank)
    model = DDP(model, device_ids=[rank])

    # 数据加载器
    dataset = QuadraticDataset()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    print("11111")

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(10):  # 训练10个epoch
        model.train()
        sampler.set_epoch(epoch)
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Rank {rank}, Epoch [{epoch+1}/10], Loss: {running_loss/len(dataloader)}")

# 启动分布式训练
def main():
    print("---------")

    size = 4  # 4个GPU
    rank = int(os.environ['LOCAL_RANK'])  # 获取当前进程的rank

    print(rank)
    init_process(rank, size, train)

if __name__ == "__main__":
    main()
    # 
