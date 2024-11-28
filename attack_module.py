import numpy as np
import torch
import torch.nn.functional as F


def poison_training_data(data, target, trigger_value=255, poison_fraction=0.05):
    num_poison = int(len(data) * poison_fraction)
    poisoned_data = data.clone()
    poisoned_target = target.clone()

    # 在原本数据的基础上添加trigger
    for i in range(num_poison):
        poisoned_data[i][0][26:28, 26:28] = trigger_value  # 在图像右下角添加 trigger

    # 假设目标攻击类是 7
    target_class = 7
    poisoned_target[:num_poison] = target_class
    return poisoned_data, poisoned_target


def train_with_backdoor(model, train_loader, global_model, mu, epochs=10, lr=0.001, trigger_value=255,
                        poison_fraction=0.05, test_loader=None, output_file=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracy_list = []

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            # 对部分数据进行投毒
            data, target = poison_training_data(data, target, trigger_value, poison_fraction)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # FedProx 正则化项
            prox_term = 0.5 * mu * sum(
                (p1 - p2).norm() for p1, p2 in zip(model.parameters(), global_model.parameters()))
            loss += prox_term

            loss.backward()
            optimizer.step()

        if test_loader:
            acc = evaluate_model(model, test_loader)
            accuracy_list.append(acc)
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {acc:.2f}%')

    if output_file:
        with open(output_file, 'w') as f:
            for epoch, acc in enumerate(accuracy_list):
                f.write(f'Epoch {epoch + 1}, Accuracy: {acc:.2f}%\n')

    return accuracy_list


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy