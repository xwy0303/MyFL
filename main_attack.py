import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import get_datasets, get_data_loaders
from local_model import Net, local_train
from secure_aggregation import multi_krum
from utils import test
from attack_module import FedIMP
import time

if __name__ == '__main__':
    # 定义超参数
    num_nodes = 10
    attack_nodes = 4
    batch_size = 64
    learning_rate = 0.01
    epochs = 100

    # 获取数据集
    mnist_train_split, mnist_test = get_datasets()

    # 创建数据加载器列表
    train_loaders = get_data_loaders(mnist_train_split, batch_size)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)  # 干净的测试数据加载器

    # 初始化存储训练损失和准确率的列表
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    # 创建DataFrame来存储准确率数据
    acc_data = {
        'Epoch': [],
        'Accuracy': []
    }
    acc_df = pd.DataFrame(acc_data)

    # 在外面初始化FedIMP类实例，便于复用相关状态（如果有）
    fedimp = FedIMP()

    for epoch in range(1, epochs + 1):
        epoch_train_losses = []
        epoch_train_accuracies = []

        # 良性成员节点训练模型列表
        local_models = [Net() for _ in range(num_nodes - attack_nodes)]
        model = Net()
        participating_models = []
        for i in range(num_nodes - attack_nodes):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            local_train(model, train_loaders[i], optimizer, epoch, i, epoch_train_losses, epoch_train_accuracies)
            local_models.append(model)

        # 将每个epoch的损失和准确率添加到总列表中
        train_losses.extend(epoch_train_losses)
        train_accuracies.extend(epoch_train_accuracies)

        # 投毒攻击
        fedimp = FedIMP()
        mean_update, std_update = fedimp.calcMeanAndStd(local_models)
        mean_model = multi_krum(local_models, attack_nodes, 3)
        fisher_informations = []
        size_pois_dataset = []

        for i in range(attack_nodes):
            fisher_info = fedimp.calcFisherInfo(mean_model, train_loaders[i])
            fisher_informations.append(fisher_info)
            size_pois_dataset.append(len(train_loaders[i].dataset))

        avg_fisher = fedimp.weightAvgFisher(fisher_informations, size_pois_dataset)
        binary_mask = fedimp.createBinaryMask(avg_fisher)

        delta = fedimp.binarySearchBoostingCoefficient(mean_update, std_update, binary_mask, local_models)
        print("delta:{}".format(delta))

        malicious_update = fedimp.genMaliciousUpdate(mean_update, std_update, binary_mask, delta)
        model = Net()
        fedimp.create_model_from_update(model, malicious_update)

        for i in range(attack_nodes):
            local_models.append(model)

        # 使用干净数据集验证局部模型
        for i in range(num_nodes):
            local_accuracy = test(local_models[i], test_loader)
            print(f"Local model {i} accuracy on clean test set: {local_accuracy}%")

            # 如果准确率大于30%，则允许模型参与聚合
            if local_accuracy > 30:
                participating_models.append(local_models[i])

        # 全局模型聚合
        start_time = time.time()  # 记录聚合开始时间
        global_model = multi_krum(participating_models, attack_nodes,3)
        end_time = time.time()  # 记录聚合结束时间
        print(f"Multi-Krum聚合耗时: {end_time - start_time}秒")

        # 测试全局模型
        global_accuracy = test(global_model, test_loader)
        print("epoch: {}, acc: {}".format(epoch, global_accuracy))
        test_accuracies.append(global_accuracy)

        # 将当前轮的准确率添加到DataFrame中
        new_row = pd.DataFrame({'Epoch': [epoch], 'Accuracy': [global_accuracy]})
        acc_df = pd.concat([acc_df, new_row], ignore_index=True)

    # 绘制训练损失和准确率图表
    plt.figure(figsize=(12, 5))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(0, 3010)
    plt.ylim(0, 100)

    # 绘制训练准确率
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # 绘制测试准确率
    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.show()

    # 保存DataFrame为Excel文件
    acc_df.to_excel('acc_with_attack_model.xlsx', index=False)

    # 保存损失DataFrame为csv文件
    # loss_df.to_csv('loss_no_attack.csv', index=False)
