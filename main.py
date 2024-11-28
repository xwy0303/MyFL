from data_loader import get_data_loaders
from clustering import cluster_nodes, select_leader_nodes
from reputation import BayesianReputationSystem
from secure_aggregation import secure_aggregation
from resnet_model import ResNet18
from sybil_attack import train_with_sybil_attack, evaluate_model  # 使用Sybil攻击模型替换
from fedadam import train_fedadam
from weighted_majority import WeightedMajorityAlgorithm
from reputation_based_aggregation import ReputationBasedAggregation
import torch
import os
import numpy as np
import random

def calculate_similarity(param1, param2):
    """
    计算两个模型参数之间的相似度，使用L2 norm。
    """
    return sum((p1 - p2).norm().item() for p1, p2 in zip(param1, param2))

def update_reputation(reputation_system, node_id, similarity, threshold=0.5):
    """
    根据相似度阈值更新节点的信誉值。
    """
    is_consistent = similarity > threshold
    reputation_system.update_reputation(node_id, success=is_consistent)

def leader_evaluate_and_update(leader_model, nodes, reputation_system, threshold=0.5):
    """
    领导节点评估和更新成员节点的信誉值。
    """
    for node_id, node_model in enumerate(nodes):
        if node_model is leader_model:
            continue
        similarity = calculate_similarity(leader_model.parameters(), node_model.parameters())
        is_consistent = similarity > threshold
        reputation_system.update_reputation(node_id, success=is_consistent)

def filter_low_reputation_nodes(nodes, reputation_system, threshold=0.7):
    """
    过滤掉拥有低信誉值的节点。
    """
    valid_nodes = {node_id: node for node_id, node in nodes.items()
                   if reputation_system.get_reputation(node_id) > threshold}
    return valid_nodes

def get_output_file(base_file, seq_num):
    """
    生成带序列号的文件名。
    """
    return f'{base_file}_seq{seq_num:02d}.txt'

def main():
    os.environ["OMP_NUM_THREADS"] = "4"

    # 超参数设定
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10  # 仅运行10个epochs并停止
    mu = 0.01  # FedProx 参数

    if not os.path.exists('result'):
        os.makedirs('result')

    base_file = f'result/epochs{num_epochs}_lr{learning_rate}_mu{mu}_batch{batch_size}'

    # 加载训练和测试数据
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # 初始化节点数据和计算性能
    node_data = torch.randn(1000, 32 * 32 * 3).numpy()  # 模拟CIFAR-10节点数据 (3通道 * 32 * 32)
    compute_powers = np.random.rand(1000)  # 模拟节点的计算能力

    # 聚类节点并选择每个聚类的领导节点
    labels, centers = cluster_nodes(node_data, 10, compute_powers)
    leader_nodes = select_leader_nodes(labels, compute_powers)
    print(f'Leader nodes: {leader_nodes}')

    # 初始化信誉系统、加权算法和聚合机制
    reputation_system = BayesianReputationSystem()
    wma = WeightedMajorityAlgorithm()
    ram = ReputationBasedAggregation()

    # 初始化全局模型和节点模型
    global_model = ResNet18(weights=None)
    nodes = [ResNet18(weights=None) for _ in range(10)]

    all_models_params = []

    # 设定 Sybil 攻击节点
    num_nodes = len(nodes)
    num_sybil = int(0 * num_nodes)  # Sybil 攻击节点占 50%
    sybil_node_indices = random.sample(range(num_nodes), num_sybil)  # 随机选择 Sybil 节点

    seq_num = 1
    output_file = get_output_file(base_file, seq_num)

    # 对每个节点进行训练
    for node_id, node_model in enumerate(nodes):
        if node_id in sybil_node_indices:
            # Sybil 攻击训练
            print(f"Training Sybil node {node_id} with poison_fraction=0.5")
            accuracy_list = train_with_sybil_attack(node_model, train_loader, global_model, mu=mu, epochs=num_epochs,
                                                    lr=learning_rate,
                                                    test_loader=test_loader, output_file=output_file,
                                                    poison_fraction=0)
        else:
            # 正常节点训练
            accuracy_list = train_fedadam(node_model, train_loader, global_model, mu=mu, epochs=num_epochs,
                                          lr=learning_rate,
                                          test_loader=test_loader, output_file=output_file)

        # 更新节点信誉
        reputation_system.update_reputation(node_id, success=True)
        # 保存模型参数
        all_models_params.append([param.clone().detach() for param in node_model.parameters()])

    # 第一次聚类簇内模型聚合并评估信誉
    for label_id in leader_nodes:
        leader_id = leader_nodes[label_id]
        leader_model = nodes[leader_id]

        cluster_node_ids = [i for i, lbl in enumerate(labels) if lbl == label_id and i != leader_id]
        cluster_models = {i: nodes[i] for i in cluster_node_ids}

        # 聚合成员模型生成初步聚类簇内模型
        initial_aggregated_model = ram.aggregate_models(cluster_models)
        print(f"Initial Aggregated Model for Cluster {label_id}:")

        # 计算成员节点与聚类簇内模型之间的相似度，更新信誉
        for node_id in cluster_models:
            similarity = calculate_similarity(initial_aggregated_model.parameters(), nodes[node_id].parameters())
            update_reputation(reputation_system, node_id, similarity)

    # 聚合所有领导节点
    leader_models = {idx: nodes[leader_nodes[idx]] for idx in leader_nodes}
    global_aggregated_model = ram.aggregate_models(leader_models)
    print("Global Aggregated Model after first round of aggregation:")

    # 评估领导节点和全局模型的差异，设置领导节点的信誉值
    for leader_id in leader_nodes.values():
        similarity = calculate_similarity(global_aggregated_model.parameters(), nodes[leader_id].parameters())
        update_reputation(reputation_system, leader_id, similarity)

    # 广播全局模型到领导节点，并评估成员节点与全局模型的相似度，第二次设置成员节点信誉值
    for label_id in leader_nodes:
        leader_id = leader_nodes[label_id]
        cluster_node_ids = [i for i, lbl in enumerate(labels) if lbl == label_id]

        for node_id in cluster_node_ids:
            similarity = calculate_similarity(global_aggregated_model.parameters(), nodes[node_id].parameters())
            update_reputation(reputation_system, node_id, similarity)

    # 过滤信誉低的节点
    valid_nodes = filter_low_reputation_nodes(nodes, reputation_system, 0.7)
    print(f"Valid nodes after filtering (final): {valid_nodes.keys()}")

    # 生成最终的全局模型
    final_aggregated_model = ram.aggregate_models(valid_nodes)
    print("Final Aggregated Model:")

    # 评估最终全局模型在测试集上的准确率
    accuracy = evaluate_model(final_aggregated_model, test_loader)
    print(f"Final Aggregated Model Accuracy: {accuracy:.2f}%")

    # 保存结果到文件
    output_file = get_output_file(base_file, seq_num)
    with open(output_file, 'w') as f:
        f.write(f'Main Epoch: {seq_num}, Final Aggregated Model Accuracy: {accuracy:.2f}%\n')

if __name__ == "__main__":
    main()