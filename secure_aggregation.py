import torch
import numpy as np
from local_model import Net


def euclidean_distance(x, y):
    # 将OrderedDict转换为张量
    x_tensor = torch.cat([v.view(-1) for v in x.values()])
    y_tensor = torch.cat([v.view(-1) for v in y.values()])
    # 计算并返回欧几里得距离
    return np.linalg.norm(x_tensor.cpu().numpy() - y_tensor.cpu().numpy())


def secure_aggregation(models, n_attackers):
    aggregated_model = Net()  # 这里假设Net类在local_model.py中已经定义
    for param in aggregated_model.parameters():
        param.data.zero_()

    num_clients = len(models)
    dist_matrix = np.zeros((num_clients, num_clients))

    # 计算权重之间的距离
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = euclidean_distance(models[i].state_dict(), models[j].state_dict())
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 计算每个参与者的距离和，并选择距离和最小的模型
    min_sum_dist = float('inf')
    selected_indices = []
    for i in range(num_clients):
        sorted_indices = np.argsort(dist_matrix[i])
        sum_dist = np.sum(dist_matrix[i, sorted_indices[1:(n_attackers + 1)]])
        if sum_dist < min_sum_dist:
            min_sum_dist = sum_dist
            selected_indices = [sorted_indices[0]]
        elif sum_dist == min_sum_dist:
            selected_indices.append(sorted_indices[0])

    # 聚合选中模型的参数
    for param, selected_param in zip(aggregated_model.parameters(), models[selected_indices[0]].parameters()):
        param.data.copy_(selected_param.data)

    return aggregated_model