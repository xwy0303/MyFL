import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import get_datasets, get_data_loaders
from local_model import Net, local_train
from fedavg import fedavg, client_update
from utils import test
# from secure_aggregation import
from attack_model import CMP

if __name__ == '__main__':
    # 定义超参数
    num_nodes = 10
    batch_size = 64
    learning_rate = 0.01
    epochs = 100
    local_epochs = 1  # 每个客户端在通信轮次之间执行的本地训练轮数

    # 获取数据集
    mnist_train_split, mnist_test = get_datasets()

    # 创建数据加载器列表
    train_loaders = get_data_loaders(mnist_train_split, batch_size)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    # 初始化存储训练损失和准确率的列表
    train_losses = [[] for _ in range(num_nodes)]  # 每个客户端的损失列表
    train_accuracies = [[] for _ in range(num_nodes)]  # 每个客户端的准确率列表
    test_accuracies = []

    # 创建DataFrame来存储准确率数据
    acc_data = {
        'Epoch': [],
        'Accuracy': []
    }
    acc_df = pd.DataFrame(acc_data)

    # 迭代执行联邦平均
    for epoch in range(1, epochs + 1):
        # 清空之前的模型和权重列表
        models = []
        weights = []

        # 添加正常模型并训练
        for i in range(num_nodes):
            model = Net()  # 实例化模型
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            # 使用 local_train 函数训练模型
            local_train(model, train_loaders[i], optimizer, epoch, i, train_losses[i], train_accuracies[i])
            models.append(model)
            weights.append(1.0 / num_nodes)  # 假设每个客户端的权重相等

        # ...[投毒攻击代码]...

        # 聚合模型
        global_model = fedavg(models, weights)

        # 使用 client_update 更新客户端模型
        for i in range(num_nodes):
            model = global_model  # 使用当前全局模型初始化
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            client_update(model, train_loaders[i], optimizer, local_epochs, batch_size)

        # 测试全局模型
        global_accuracy = test(global_model, test_loader)
        print("Epoch: {}, Global Accuracy: {}".format(epoch, global_accuracy))
        test_accuracies.append(global_accuracy)

        # 将当前轮的准确率添加到DataFrame中
        new_row = pd.DataFrame({'Epoch': [epoch], 'Accuracy': [global_accuracy]})
        acc_df = pd.concat([acc_df, new_row], ignore_index=True)

    # 绘制测试准确率图表
    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, epochs)
    plt.ylim(0, 100)

    plt.show()

    # 保存DataFrame为Excel文件
    acc_df.to_excel('acc_with_fedavg.xlsx', index=False)