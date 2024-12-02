import torch
import numpy as np
from local_model import Net


def trimmed_mean_aggregation(models, num_malicious):
    if not models:
        raise ValueError("模型列表不能为空。")

    num_clients = len(models)
    if num_malicious >= num_clients / 2:
        raise ValueError("恶意客户端数量不能超过客户端总数的一半。")

    # 计算修剪比例
    trim_ratio = min(num_malicious / num_clients, 0.5)

    # 计算要修剪的数量
    num_to_trim = int(num_clients * trim_ratio)
    if num_to_trim < 1:
        num_to_trim = 1

    aggregated_model = Net()
    for param in aggregated_model.parameters():
        param.data.zero_()

    # 对每个参数进行处理
    for param_name in aggregated_model.state_dict().keys():
        param_values = [model.state_dict()[param_name].view(-1) for model in models]

        # 将参数值堆叠成一个矩阵
        param_matrix = torch.stack(param_values, dim=0)

        # 按照每个元素进行排序
        sorted_param_matrix, _ = torch.sort(param_matrix, dim=0)

        # 去除最大和最小的num_to_trim个值
        trimmed_param_matrix = sorted_param_matrix[num_to_trim:-num_to_trim, :]

        # 计算剩余值的平均值
        aggregated_value = trimmed_param_matrix.mean(dim=0)

        # 更新聚合模型的参数
        aggregated_model.state_dict()[param_name].copy_(
            aggregated_value.view(aggregated_model.state_dict()[param_name].size()))

    return aggregated_model