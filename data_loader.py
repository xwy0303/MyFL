import torch
import torchvision
import torchvision.transforms as transforms

def get_datasets():
    # 定义数据转换
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # 获取FMNIST训练集和测试集
    fmnist_train = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    fmnist_test = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # 获取MNIST训练集和测试集
    mnist_train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return fmnist_train, fmnist_test, mnist_train, mnist_test