import torch
from local_model import Net

def secure_aggregation(models):
    aggregated_model = Net()  # 这里假设Net类在local_model.py中已经定义
    for param in aggregated_model.parameters():
        param.data.zero_()
    for model in models:
        for param, aggregated_param in zip(model.parameters(), aggregated_model.parameters()):
            aggregated_param.data += param.data.clone() / len(models)
    return aggregated_model