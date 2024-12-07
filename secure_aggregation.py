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
    x_tensor = torch.cat([v.view(-1) for v in x.values()], dim=0)
    y_tensor = torch.cat([v.view(-1) for v in y.values()], dim=0)
    return torch.linalg.norm(x_tensor - y_tensor).item()

def build_distance_matrix(models):
    num_clients = len(models)
    model_tensors = [torch.cat([v.view(-1) for v in model.state_dict().values()], dim=0) for model in models]
    model_tensors = torch.stack(model_tensors)
    i_indices, j_indices = np.triu_indices(num_clients, 1)
    dist_matrix = torch.zeros((num_clients, num_clients))
    for i, j in zip(i_indices, j_indices):
        dist = euclidean_distance(models[i].state_dict(), models[j].state_dict())
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
    return dist_matrix

def calculate_scores(dist_matrix, f):
    num_clients = dist_matrix.size(0)
    # 转置距离矩阵以优化排序和求和操作
    transposed_dist_matrix = dist_matrix.t()
    # 对转置后的矩阵列进行排序，获取排序后的索引矩阵
    sorted_indices_matrix = torch.argsort(transposed_dist_matrix, dim=1)
    scores = []

    for i in range(num_clients):
        # 针对f为0的情况单独处理，直接将自身到自身距离（为0）作为分数
        if f == 0:
            scores.append(0.0)
            continue

        end_index = min(f + 1, num_clients)
        # 修正切片起始索引从0开始，这样能正确取到f个近邻（不含自身）的距离进行求和
        sum_dist = torch.sum(transposed_dist_matrix[i, sorted_indices_matrix[i, 0:end_index - 1]])
        scores.append(sum_dist.item())

    return scores

def select_models(scores, m):
    selected_indices = np.argsort(scores)[:m]
    return selected_indices

def multi_krum(models, f, m, distance_threshold=None):
    # 检查输入有效性
    if not models:
        raise ValueError("模型列表不能为空。")
    if f >= len(models):
        raise ValueError("拜占庭攻击者数量不能大于或等于客户端数量。")
    if m > len(models) - f or m <= 0:
        raise ValueError("m 的取值应满足 0 < m <= n - f。")

    aggregated_model = Net()
    for param in aggregated_model.parameters():
        param.data.zero_()

    dist_matrix = build_distance_matrix(models)
    # 增加对模型参数合法性的检查（这里简单示例检查是否有NaN值）
    for model in models:
        for param in model.parameters():
            if torch.isnan(param).any():
                raise ValueError("模型参数中存在非法的NaN值")

    scores = calculate_scores(dist_matrix, f)
    selected_indices = select_models(scores, m)
    selected_models = [models[i] for i in selected_indices]
    with torch.no_grad():  # 不需要计算梯度，提高效率并避免不必要的计算图构建
        for target_param, source_params in zip(aggregated_model.parameters(), zip(*[s_model.parameters() for s_model in selected_models])):
            target_param.data.copy_(torch.stack(source_params).sum(dim=0) / m)

    return aggregated_model