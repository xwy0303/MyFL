import numpy as np
from sklearn.cluster import KMeans

def cluster_nodes(data, n_clusters, compute_powers):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++')
    kmeans.fit(data)
    return kmeans.labels_, kmeans.cluster_centers_

def select_leader_nodes(labels, compute_powers):
    unique_labels = np.unique(labels)
    leader_nodes = {}
    for label in unique_labels:
        indices = np.where(labels == label)[0]
        max_compute_index = indices[np.argmax([compute_powers[i] for i in indices])]
        leader_nodes[label] = max_compute_index
    return leader_nodes

if __name__ == "__main__":
    node_data = np.random.rand(1000, 784)
    compute_powers = np.random.rand(1000)  # 假设每个节点的算力为0到1之间的随机数
    labels, centers = cluster_nodes(node_data, 10, compute_powers)
    leader_nodes = select_leader_nodes(labels, compute_powers)
    print(f'Leader nodes: {leader_nodes}')