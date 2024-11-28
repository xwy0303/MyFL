class WeightedMajorityAlgorithm:
    def __init__(self, initial_weight=1.0):
        self.weights = {}
        self.initial_weight = initial_weight

    def update_weights(self, node_id, is_consistent):
        if node_id not in his.weights:
            self.weights[node_id] = self.initial_weight
        if is_consistent:
            self.weights[node_id] *= 1.0
        else:
            self.weights[node_id] *= 0.5

    def get_weights(self):
        return self.weights

if __name__ == "__main__":
    wma = WeightedMajorityAlgorithm()
    wma.update_weights(node_id=1, is_consistent=True)
    print(f'Node 1 Weight: {wma.get_weights()}')