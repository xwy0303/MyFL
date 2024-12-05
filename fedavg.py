import torch
import numpy as np


def euclidean_distance(x, y):
    """
    计算两个模型参数之间的欧几里得距离。

    参数：
    x (OrderedDict)：第一个模型的参数。
    y (OrderedDict)：第二个模型的参数。

    返回：
    float：欧几里得距离。
    """
    x_tensor = torch.cat([v.view(-1) for v in x.values()])
    y_tensor = torch.cat([v.view(-1) for v in y.values()])
    return np.linalg.norm(x_tensor.cpu().numpy() - y_tensor.cpu().numpy())


def fedavg(models, weights):
    """
    执行联邦平均聚合操作。

    参数：
    models (list)：包含客户端模型的列表。
    weights (list)：每个模型的权重，用于加权平均。

    返回：
    aggregated_model (Net): 聚合后的模型。
    """
    if not models:
        raise ValueError("模型列表不能为空。")

    num_models = len(models)
    if len(weights) != num_models:
        raise ValueError("权重列表长度必须与模型列表长度匹配。")

    # 创建一个新的模型实例，用于存储聚合后的模型参数
    aggregated_model = models[0]  # 假设所有模型结构相同，取第一个模型作为基础
    aggregated_model_state_dict = aggregated_model.state_dict()

    # 初始化聚合后的模型参数为零，确保是浮点数类型
    for key in aggregated_model_state_dict.keys():
        aggregated_model_state_dict[key] = torch.zeros_like(aggregated_model_state_dict[key], dtype=torch.float32)

    # 根据权重聚合模型参数
    for model, weight in zip(models, weights):
        model_state_dict = model.state_dict()
        with torch.no_grad():
            for key in aggregated_model_state_dict.keys():
                # 确保权重和模型参数都是浮点数类型
                aggregated_model_state_dict[key] += weight * model_state_dict[key].float()

    # 归一化聚合后的参数
    for key in aggregated_model_state_dict.keys():
        aggregated_model_state_dict[key] /= sum(weights)

    # 加载聚合后的参数到新模型
    aggregated_model.load_state_dict(aggregated_model_state_dict)
    return aggregated_model


def client_update(model, dataset, optimizer, local_epochs, batch_size):
    """
    在客户端执行本地更新。

    参数：
    model (Net): 要更新的模型。
    dataset (Dataset): 客户端的数据集。
    optimizer (Optimizer): 优化器。
    epochs (int): 本地训练的轮数。
    batch_size (int): 批处理大小。

    返回：
    model (Net): 更新后的模型。
    """
    model.train()

    for epoch in range(local_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataset):
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1},  Loss: {total_loss / len(dataset)}')
    return model