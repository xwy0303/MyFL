import torch
import numpy as np
from local_model import Net

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

def secure_aggregation(models, n_attackers):
    """
    执行安全聚合操作，类似于Krum算法。

    参数：
    models (list)：包含客户端模型的列表。
    n_attackers (int)：攻击者的数量。

    返回：
    Net：聚合后的模型。
    """
    # 检查输入有效性
    if not models:
        raise ValueError("模型列表不能为空。")
    if n_attackers >= len(models):
        raise ValueError("攻击者数量不能大于或等于客户端数量。")

    num_clients = len(models)
    aggregated_model = Net()
    for param in aggregated_model.parameters():
        param.data.zero_()

    # 一次性将模型参数转换为张量并存储
    model_tensors = [torch.cat([v.view(-1) for v in model.state_dict().values()]) for model in models]

    # 计算距离矩阵
    dist_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = euclidean_distance(model_tensors[i], model_tensors[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 计算每个参与者的距离和，并选择距离和最小的模型
    min_sum_dists = [float('inf')] * num_clients
    for i in range(num_clients):
        sum_dist = np.sum(dist_matrix[i])
        min_sum_dists[i] = sum_dist

    # 选择距离和最小的模型（这里选择n_attackers+1个，根据Krum算法）
    selected_indices = np.argsort(min_sum_dists)[n_attackers:]

    # 聚合选中模型的参数
    for param in aggregated_model.parameters():
        param.data.zero_()
    for i in selected_indices:
        for param, selected_param in zip(aggregated_model.parameters(), models[i].parameters()):
            param.data += selected_param.data
    for param in aggregated_model.parameters():
        param.data /= len(selected_indices)

    return aggregated_model