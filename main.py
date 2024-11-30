# main.py
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from data_loader import get_datasets, get_data_loaders
from local_model import Net, local_train
from secure_aggregation import secure_aggregation
from utils import test
from attack_model import CMP  # 导入新的攻击模型

if __name__ == '__main__':
    # 定义超参数
    num_nodes = 10
    batch_size = 64
    learning_rate = 0.01
    epochs = 5

    # 获取数据集
    mnist_train_split, mnist_test = get_datasets()

    # 创建数据加载器列表
    train_loaders = get_data_loaders(mnist_train_split, batch_size)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    # 初始化存储训练损失和准确率的列表
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(1, epochs + 1):
        epoch_train_losses = []
        epoch_train_accuracies = []

        # 成员节点训练模型列表
        local_models = []
        for i in range(num_nodes):
            model = Net()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            local_train(model, train_loaders[i], optimizer, epoch, i, epoch_train_losses, epoch_train_accuracies)
            local_models.append(model)

        # 将每个epoch的损失和准确率添加到总列表中
        train_losses.extend(epoch_train_losses)
        train_accuracies.extend(epoch_train_accuracies)

        # 投毒攻击
        num_malicious = 3  # 假设有3个恶意节点
        cmp = CMP(num_malicious)
        malicious_model = Net()
        malicious_model = cmp.create_model_from_update(malicious_model, None)

        for _ in range(num_malicious):
            local_models.append(malicious_model)

        # 全局模型聚合
        global_model = secure_aggregation(local_models)

        # 测试全局模型
        global_accuracy = test(global_model, test_loader)
        print("epoch: {}, acc: {}".format(epoch, global_accuracy))
        test_accuracies.append(global_accuracy)

    # 绘制训练损失和准确率图表
    plt.figure(figsize=(12, 5))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制训练准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()

    # 绘制测试准确率
    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()