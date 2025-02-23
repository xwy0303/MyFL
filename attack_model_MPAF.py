import torch
import torch.nn as nn

class MPAF:
    def __init__(self, base_model, scale_factor=1e6):
        self.base_model = base_model
        self.scale_factor = scale_factor

    def MPAF(self, model):
        """
        在本地模型更新上添加攻击扰动
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # 计算从当前模型参数到基模型参数的差值，并按比例放大
                    param.grad += self.scale_factor * (self.base_model.state_dict()[name] - param.data).detach()