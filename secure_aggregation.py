import torch
import numpy as np
from local_model import Net
import heapq

def euclidean_distance(x, y):
    x_tensor = torch.cat([v.view(-1) for v in x.values()])
    y_tensor = torch.cat([v.view(-1) for v in y.values()])
    return np.linalg.norm(x_tensor.cpu().numpy() - y_tensor.cpu().numpy())

def secure_aggregation(models, n_attackers):
    if not models:
        raise ValueError("模型列表不能为空。")
    num_clients = len(models)
    if n_attackers >= num_clients - 2:
        raise ValueError("攻击者数量不能大于等于客户端总数减去2。")

    # 计算每个客户端到其他客户端的距离
    dist_matrix = np.zeros((num_clients, num_clients))
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            dist = euclidean_distance(models[i].state_dict(), models[j].state_dict())
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # 为每个客户端计算所有其他客户端的距离之和，去掉最大的n_attackers+1个距离
    k = num_clients - n_attackers - 2
    if k <= 0:
        raise ValueError("没有足够的非攻击者客户端来执行Krum聚合。")

    sum_dists = []
    for i in range(num_clients):
        sorted_indices = np.argsort(dist_matrix[i])
        sum_dist = np.sum(dist_matrix[i, sorted_indices[1:k+1]])  # 排除自身，取k个最小距离
        sum_dists.append((sum_dist, i))

    # 选择距离之和最小的客户端
    _, selected_index = min(sum_dists, key=lambda x: x[0])

    # 返回被选中的客户端的模型
    selected_model = models[selected_index]
    aggregated_model = Net()
    aggregated_model.load_state_dict(selected_model.state_dict())
    return aggregated_model

# 使用示例
# models = [Net() for _ in range(10)]  # 假设这里有10个客户端模型
# n_attackers = 2
# secure_model = krum_aggregation(models, n_attackers)