import torch
import numpy as np
import torch.nn.functional as F


def poison_data_with_sybil(model, data, target, trigger_value=255, poison_fraction=0.05):
    """
    对部分训练数据进行 Sybil 攻击（输入添加 trigger 并篡改标签）。
    """
    num_poison = int(len(data) * poison_fraction)
    poisoned_data = data.clone()
    poisoned_target = target.clone()

    # 在图像右下角添加 trigger，模拟投毒输入。
    for i in range(num_poison):
        poisoned_data[i][:, 30:32, 30:32] = trigger_value
        # Note: CIFAR-10 是 3 通道图像，因此选取最后两个像素位置

    # 攻击目标类别（假设选择类别 7）
    target_class = 7
    poisoned_target[:num_poison] = target_class

    # 打印部分投毒数据样本和标签，用于调试和验证
    print("Poisoned data sample shape:", poisoned_data[:num_poison][0].shape)
    print("Poisoned data sample (flattened):", poisoned_data[:num_poison][0].flatten()[:100])  # 仅打印部分数据
    print("Poisoned target label:", poisoned_target[:num_poison][0])

    return poisoned_data, poisoned_target


def train_with_sybil_attack(model, train_loader, global_model, mu, epochs=10, lr=0.001, trigger_value=255,
                            poison_fraction=0.05, sybil_nodes=None, test_loader=None, output_file=None):
    """
    带有 Sybil 攻击的训练函数。
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    accuracy_list = []

    attacked_samples_logged = False  # 用于调试，确保仅打印一次攻击样本

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = poison_data_with_sybil(model, data, target, trigger_value, poison_fraction)

            if not attacked_samples_logged:
                # 打印调试信息，确保投毒样本正确生成
                print("DEBUG - Sybil Attack - Poison Data & Target:")
                print("Data shape:", data.shape)
                print("Sample data:", data[0].flatten()[:10])  # 打印部分数据
                print("Sample target:", target[:10])
                attacked_samples_logged = True

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # FedProx 正则化项
            prox_term = 0.5 * mu * sum(
                (p1 - p2).norm() for p1, p2 in zip(model.parameters(), global_model.parameters()))
            loss += prox_term

            loss.backward()
            optimizer.step()

        # 评估每个测试epoch 输出准确率
        if test_loader:
            acc = evaluate_model(model, test_loader)
            accuracy_list.append(acc)
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {acc:.2f}%')

    # 记录准确率输出到文件
    if output_file:
        with open(output_file, 'w') as f:
            for epoch, acc in enumerate(accuracy_list):
                f.write(f'Epoch {epoch + 1}, Accuracy: {acc:.2f}%\n')

    return accuracy_list


def evaluate_model(model, test_loader):
    """
    评估模型在测试或验证数据集上的表现。
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