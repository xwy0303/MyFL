import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import pandas as pd

from attack_model_MPAF import MPAF
from data_loader import get_datasets, get_data_loaders
from local_model import Net, local_train
from fedavg import fedavg, client_update
from utils import test
import time

if __name__ == '__main__':
    num_nodes = 3
    batch_size = 64
    learning_rate = 0.01
    epochs = 100
    local_epochs = 1

    mnist_train_split, mnist_test = get_datasets()
    train_loaders = get_data_loaders(mnist_train_split, batch_size)
    test_loader = data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    loss_data = {'Epoch': [], 'Loss': []}
    loss_df = pd.DataFrame(loss_data)

    acc_data = {'Epoch': [], 'Accuracy': []}
    acc_df = pd.DataFrame(acc_data)

    base_model = Net()  # 创建基模型
    attack = MPAF(base_model, scale_factor=1e6)  # 初始化MPAF攻击模型

    for epoch in range(1, epochs + 1):
        epoch_train_losses = []
        epoch_train_accuracies = []

        local_models = []
        weights = []
        model = Net()

        for i in range(10 - num_nodes):
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            local_train(model, train_loaders[i], optimizer, epoch, i, epoch_train_losses, epoch_train_accuracies, attack)
            local_models.append(model)
            weights.append(1.0 / num_nodes)

        train_losses.extend(epoch_train_losses)
        train_accuracies.extend(epoch_train_accuracies)

        avg_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0

        start_time = time.time()
        global_model = fedavg(local_models, weights)
        end_time = time.time()
        print(f"fedavg聚合耗时: {end_time - start_time}秒")

        for i in range(10 - num_nodes):
            model = global_model
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            client_update(model, train_loaders[i], optimizer, local_epochs, batch_size)

        end_time = time.time()
        print(f"client_update耗时: {end_time - start_time}秒")

        global_accuracy = test(global_model, test_loader)
        print("epoch: {}, acc: {}".format(epoch, global_accuracy))
        test_accuracies.append(global_accuracy)

        new_loss_row = pd.DataFrame({'Epoch': [epoch], 'Loss': [avg_loss]})
        loss_df = pd.concat([loss_df, new_loss_row], ignore_index=True)

        new_acc_row = pd.DataFrame({'Epoch': [epoch], 'Accuracy': [global_accuracy]})
        acc_df = pd.concat([acc_df, new_acc_row], ignore_index=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.figure()
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    plt.show()

    acc_df.to_excel('acc_with_attack_model.xlsx', index=False)
    loss_df.to_csv('loss_with_attack_model.csv', index=False)