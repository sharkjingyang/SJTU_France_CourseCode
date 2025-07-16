import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 定义ToyModel
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

# 获取数据集
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    trainloader = torch.utils.data.DataLoader(my_trainset, batch_size=16, num_workers=2, sampler=train_sampler)
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

# 定义优化器
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
        if dist.get_rank() == 0:
            iterator.set_description(f"Epoch {epoch}, Loss: {loss.item():.3f}")
