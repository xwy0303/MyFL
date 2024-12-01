import torch
import torchvision
import torchvision.transforms as transforms

def get_datasets():
    # 定义数据转换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 获取MNIST训练集和测试集
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 计算每个划分的大小
    total_size = len(fmnist_train)
    split_size = total_size // 10

    # 将MNIST训练集划分成10份
    fmnist_train_split = torch.utils.data.random_split(fmnist_train, [split_size] * 10)

    return fmnist_train_split, fmnist_test

def get_data_loaders(fmnist_train_split, batch_size):
    train_loaders = []
    for i in range(10):
        train_loader = torch.utils.data.DataLoader(fmnist_train_split[i], batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    return train_loaders