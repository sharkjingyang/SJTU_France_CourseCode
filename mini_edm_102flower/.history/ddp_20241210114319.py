import os
import torch
import torch.distributed as dist
from torch.utils.data import dataloader,Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# 获取 local_rank
local_rank = int(os.environ['LOCAL_RANK'])

# 设置当前 GPU
torch.cuda.set_device(local_rank)

# 初始化分布式进程组
dist.init_process_group(
    backend='nccl',  # 使用 NCCL 作为后端，适合 GPU 分布式训练
    rank=local_rank,  # 当前进程的 rank
    world_size=torch.cuda.device_count()  # 总进程数（即总 GPU 数）
)

# 定义 ToyModel
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QuadraticDataset(Dataset):
    def __init__(self, start=-10, end=10, num_samples=1000):
        # 生成从 start 到 end 的 num_samples 个样本
        self.x = np.linspace(start, end, num_samples)  # 生成 x 值
        self.y = self.x ** 2  # 对应的 y 值为 x^2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # 返回一对 (x, y)
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
    
# 获取数据集
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=16, num_workers=2, sampler=train_sampler)

    dataset = QuadraticDataset(start=-10, end=10, num_samples=1000)
    
    # 使用 DistributedSampler 进行分布式训练
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    
    # 创建 DataLoader
    trainloader = DataLoader(dataset, batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader

trainloader = get_dataset()
model = ToyModel().to('cuda')
model = DDP(model, device_ids=[local_rank])

# 定义优化器和损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss().to('cuda')

# 训练模型
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    trainloader.sampler.set_epoch(epoch)
    for data, label in trainloader:
        data, label = data.to('cuda'), label.to('cuda')
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        optimizer.step()
        if dist.get_rank() == 0:  # 只在主进程打印
            iterator.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")

