import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from data_loader import get_datasets, get_data_loaders
from local_model import Net, local_train
from secure_aggregation import secure_aggregation
from utils import test
from attack_model import PoisonFRS

if __name__ == '__main__':
    # 定义超参数
    num_nodes = 10
    batch_size = 64
    learning_rate = 0.01
    epochs = 30

    # 获取数据集
    mnist_train_split, mnist_test = get_datasets()

    # 创建数据加载器列表
    train_loaders = get_data_loaders(mnist_train_split, batch_size)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

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

        # 创建PoisonFRS攻击实例
        # 假设我们已经有了item_embedding, target_item, popular_items, filler_items, scale_factor
        item_embedding = {0: torch.randn(10), 1: torch.randn(10)}  # 示例嵌入
        target_item = 0  # 假设目标物品是0
        popular_items = [1]  # 假设流行物品是1
        filler_items = [2]  # 假设填充物品是2
        scale_factor = 10  # 缩放因子
        poison_attack = PoisonFRS(local_models[0], item_embedding, target_item, popular_items, filler_items, scale_factor)

        # 应用攻击
        updates = poison_attack.craft_updates(epoch)
        poison_attack.apply_update(updates)

        # 全局模型聚合
        global_model = secure_aggregation(local_models, n_attackers=1)

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