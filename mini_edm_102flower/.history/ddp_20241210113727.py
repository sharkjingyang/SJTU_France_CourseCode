import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# # 获取 local_rank
# local_rank = int(os.environ['LOCAL_RANK'])

# # 设置当前 GPU
# torch.cuda.set_device(local_rank)

# # 初始化分布式进程组
# dist.init_process_group(
#     backend='nccl',  # 使用 NCCL 作为后端，适合 GPU 分布式训练
#     rank=local_rank,  # 当前进程的 rank
#     world_size=torch.cuda.device_count()  # 总进程数（即总 GPU 数）
# )

# # 定义 ToyModel
# class ToyModel(nn.Module):
#     def __init__(self):
#         super(ToyModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 获取数据集
# def get_dataset():
#     transform = torchvision.transforms.Compose([
#         torchvision.transforms.ToTensor(),
#         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
#     my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
#     train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
#     trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=16, num_workers=2, sampler=train_sampler)
#     return trainloader

# # 加载数据集
# trainloader = get_dataset()

# # 构建模型
# model = ToyModel().to('cuda')
# model = DDP(model, device_ids=[local_rank])

# # 定义优化器和损失函数
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
# loss_func = nn.CrossEntropyLoss().to('cuda')

# # 训练模型
# model.train()
# iterator = tqdm(range(100))
# for epoch in iterator:
#     trainloader.sampler.set_epoch(epoch)
#     for data, label in trainloader:
#         data, label = data.to('cuda'), label.to('cuda')
#         optimizer.zero_grad()
#         prediction = model(data)
#         loss = loss_func(prediction, label)
#         loss.backward()
#         optimizer.step()
#         if dist.get_rank() == 0:  # 只在主进程打印
#             iterator.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")


# 定义 ToyModel，用于二次回归
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)  # 输入一个数值，输出64个特征
        self.fc2 = nn.Linear(64, 64)  # 隐藏层，64个神经元
        self.fc3 = nn.Linear(64, 1)   # 输出一个数值（回归任务）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建一个简单的二次函数数据集
class QuadraticDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.x = torch.linspace(-5, 5, num_samples).view(-1, 1)  # 输入：从-5到5的1000个数值
        self.y = 2 * self.x**2 + 3 * self.x + 1  # 二次函数：y = 2x^2 + 3x + 1
        self.y += torch.randn_like(self.y) * 0.1  # 添加一些噪声

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 加载数据集
def get_dataset():
    dataset = QuadraticDataset(num_samples=1000)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    trainloader = DataLoader(dataset, batch_size=32, num_workers=2, sampler=train_sampler)
    return trainloader

# 初始化 DDP（分布式训练）
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
args = parser.parse_args()

torch.cuda.set_device(args.local_rank)

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_func = nn.MSELoss().to('cuda')

# 训练模型
model.train()
iterator = tqdm(range(100))  # 训练100个epoch
for epoch in iterator:
    trainloader.sampler.set_epoch(epoch)  # 设置分布式训练的epoch
    for data, label in trainloader:
        data, label = data.to('cuda'), label.to('cuda')
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()

        if dist.get_rank() == 0:  # 只在主进程打印loss
            iterator.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")

# 测试（可选）
if dist.get_rank() == 0:
    with torch.no_grad():
        test_data = torch.linspace(-5, 5, 100).view(-1, 1).to('cuda')
        test_output = model(test_data)
        plt.plot(test_data.cpu().numpy(), test_output.cpu().numpy(), label='Model Prediction')
        plt.plot(test_data.cpu().numpy(), (2 * test_data**2 + 3 * test_data + 1).cpu().numpy(), label='True Function', linestyle='--')
        plt.legend()
        plt.show()
