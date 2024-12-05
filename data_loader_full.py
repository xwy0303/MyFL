import torch
import torchvision
import torchvision.transforms as transforms

def get_datasets():
    # 定义数据转换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 获取 MNIST 训练集和测试集
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return mnist_train, mnist_test

def get_data_loaders(mnist_train, batch_size, num_clients):
    train_loaders = []
    for _ in range(num_clients):
        train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        train_loaders.append(train_loader)
    return train_loaders