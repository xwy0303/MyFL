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

    def create_model_from_update(self, model, update):
        """
        创建一个包含随机更新的恶意模型。

        参数：
        model (Net): 基础模型。
        update (Tensor): 模型更新。

        返回：
        malicious_model (Net): 包含随机更新的恶意模型。
        """
        malicious_state_dict = model.state_dict()
        for key in malicious_state_dict.keys():
            # 确保张量是浮点数类型
            if malicious_state_dict[key].dtype == torch.long:
                malicious_state_dict[key] = malicious_state_dict[key].float()
            # 生成与模型参数形状相同的标准正态分布随机数
            malicious_state_dict[key] = torch.randn_like(malicious_state_dict[key], dtype=torch.float32)
        model.load_state_dict(malicious_state_dict)
        return model