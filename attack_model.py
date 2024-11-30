import torch
import copy


class CMP:
    def __init__(self, num_malicious):
        self.num_malicious = num_malicious

    def attack(self, models, benign_models, learning_rate, num_iterations):
        malicious_models = [models[0]] * self.num_malicious  # 初始恶意模型
        for _ in range(num_iterations):
            # 梯度下降更新恶意模型
            for i in range(self.num_malicious):
                malicious_models[i] = self.update_model(malicious_models[i], -torch.randn_like(malicious_models[i]),
                                                        learning_rate)

            # 聚合模型
            aggregated_model = self.krum_aggregation(benign_models + malicious_models, models[0])

            # 检查是否攻击成功
            if torch.allclose(aggregated_model, malicious_models[0]):
                print("攻击成功！聚合模型被恶意模型控制。")
                return True

        print("攻击失败。")
        return False

    def update_model(self, params, gradient, learning_rate):
        return params - learning_rate * gradient

    def krum_aggregation(self, models, params):
        distances = [torch.norm(model - params) for model in models]
        min_distance_index = torch.argmin(torch.tensor(distances)).item()
        return models[min_distance_index]

    def create_model_from_update(self, model, malicious_update):
        # 创建一个与模型结构相同的随机更新
        state_dict = model.state_dict()
        malicious_state_dict = {}
        for key in state_dict.keys():
            # 生成与模型参数形状相同的随机张量
            malicious_state_dict[key] = torch.randn_like(state_dict[key])
        model.load_state_dict(malicious_state_dict)
        return model