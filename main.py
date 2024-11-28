import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from data_loader import get_datasets
from local_model import Net, local_train
from secure_aggregation import secure_aggregation
from utils import test

if __name__ == '__main__':
    # 定义超参数
    num_nodes = 5
    batch_size = 64
    learning_rate = 0.01
    epochs = 5

    # 获取数据集
    fmnist_train, fmnist_test, mnist_train, mnist_test = get_datasets()

    # 创建数据加载器列表
    train_loaders = []
    test_loaders = []
    for _ in range(num_nodes):
        # 随机分配FMNIST和MNIST数据集到成员节点（这里简单示例，实际可以根据需求更合理分配）
        if torch.randint(0, 2, (1,)).item() == 0:
            train_loader = data.DataLoader(fmnist_train, batch_size=batch_size, shuffle=True)
            test_loader = data.DataLoader(fmnist_test, batch_size=batch_size, shuffle=False)
        else:
            train_loader = data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
            test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    # 成员节点训练模型列表
    local_models = []
    for i in range(num_nodes):
        model = Net()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        train_losses = []
        train_accuracies = []
        for epoch in range(1, epochs + 1):
            local_train(model, train_loaders[i], optimizer, epoch, train_losses, train_accuracies)
        local_models.append(model)

        # 绘制成员节点训练损失和准确率图表
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss - Node {}'.format(i))
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('Training Accuracy - Node {}'.format(i))
        plt.xlabel('Batch')
        plt.ylabel('Accuracy (%)')
        plt.show()

    # 全局模型聚合
    global_model = secure_aggregation(local_models)

    # 测试全局模型
    global_accuracy = test(global_model, data.DataLoader(fmnist_test, batch_size=batch_size, shuffle=False))

    # 绘制全局模型测试损失和准确率图表（这里假设只在测试集上评估一次，若需要多次评估可修改代码）
    test_losses = []
    test_accuracies = []
    test_losses.append(test(global_model, data.DataLoader(fmnist_test, batch_size=batch_size, shuffle=False)))
    test_accuracies.append(global_accuracy)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(test_losses)
    plt.title('Global Model Test Loss')
    plt.xlabel('Evaluation')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Global Model Test Accuracy')
    plt.xlabel('Evaluation')
    plt.ylabel('Accuracy (%)')
    plt.show()