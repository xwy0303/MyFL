import torch
import numpy as np
from resnet_model import ResNet18

class ReputationBasedAggregation:
    def __init__(self):
        self.reputations = {}

    def update_reputation(self, node_id, reputation_score):
        self.reputations[node_id] = reputation_score

    def aggregate_models(self, node_models):
        total_reputation = sum(self.reputations.values())

        agg_params = {k: torch.zeros_like(v) for k, v in node_models[0].state_dict().items()}

        for node_id, model in node_models.items():
            weight = self.reputations[node_id] / total_reputation
            for name, param in model.state_dict().items():
                agg_params[name] += weight * param

        aggregated_model = ResNet18(weights=None)
        aggregated_model.load_state_dict(agg_params)

        return aggregated_model

if __name__ == "__main__":
    ram = ReputationBasedAggregation()
    ram.update_reputation(node_id=1, reputation_score=0.9)
    ram.update_reputation(node_id=2, reputation_score=0.6)

    node_models = {1: ResNet18(weights=None), 2: ResNet18(weights=None)}
    aggregated_model = ram.aggregate_models(node_models)
    print(aggregated_model)