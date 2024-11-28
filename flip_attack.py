import torch
import numpy as np
import torch.nn.functional as F


def create_adversarial_example(model, data, target, epsilon=0.2):
    """
    创建对抗样本
    """
    data.requires_grad = True

    output = model(data)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()

    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data


def train_with_flip_attack(model, train_loader, global_model, mu, epochs=10, lr=0.001, epsilon=0.1,
                           attack_fraction=0.4, test_loader=None, output_file=None):
    """
    带有FLIP攻击的训练函数
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracy_list = []

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            attack_size = int(len(data) * attack_fraction)

            # 创建对抗样本
            if attack_size > 0:
                data[:attack_size] = create_adversarial_example(model, data[:attack_size], target[:attack_size],
                                                                epsilon)

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
    """
    评估模型在测试或验证数据集上的表现
    """
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