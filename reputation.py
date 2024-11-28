class BayesianReputationSystem:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.node_reputations = {}

    def update_reputation(self, node_id, success):
        if node_id not in self.node_reputations:
            self.node_reputations[node_id] = [0, 0]

        if success:
            self.node_reputations[node_id][0] += 1
        else:
            self.node_reputations[node_id][1] += 1

        a, b = self.node_reputations[node_id]
        reputation_score = (a + self.alpha) / (a + b + self.alpha + self.beta)
        return reputation_score

    def get_reputation(self, node_id):
        if node_id not in self.node_reputations:
            return None
        a, b = self.node_reputations[node_id]
        return (a + self.alpha) / (a + b + self.alpha + self.beta)


if __name__ == "__main__":
    reputation_system = BayesianReputationSystem()
    reputation_system.update_reputation(node_id=1, success=True)
    print(f'Node 1 Reputation: {reputation_system.get_reputation(1)}')