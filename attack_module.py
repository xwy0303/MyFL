####################################################################################################
# 功能：投毒攻击
# 作者：洪建华
# 版本：创建——20241013_1350
#       编写添加触发器函数和MyAttack类的代码——20241013_1456
#       修复myattck投毒攻击无法执行的BUG——20241014_1607
#       优化数据集分配和myattack投毒训练——20241014_1634
#       优化myattack投毒训练损失函数，增加裁剪范围缩放的功能——20241101_1023
#       添加FedIMP投毒攻击——20241115_1514
####################################################################################################
import torch
import torch.nn as nn


# FedIMP攻击
class FedIMP():
    ################################################################################################
    # 功能：计算模型均值和方差
    # 输入：models：模型
    # 输出：mean_update：均值
    #       std_update：方差
    ################################################################################################
    def calcMeanAndStd(self, models):
        mean_update = {name: torch.zeros_like(param) for name, param in models[0].state_dict().items()}
        std_update = {name: torch.zeros_like(param) for name, param in models[0].state_dict().items()}

        for name in mean_update.keys():
            layer_updates = torch.stack([model.state_dict()[name].float() for model in models])
            mean_update[name] = torch.mean(layer_updates, dim=0)
            # 设置unbiased=False来避免自由度问题
            std_update[name] = torch.std(layer_updates, dim=0, unbiased=False)

        return mean_update, std_update


    ################################################################################################
    # 功能：参数重要性评估：计算Fisher信息
    # 输入：model：计算Fisher信息的模型
    #       data_loader：计算Fisher信息的数据集
    #       device：设备
    # 输出：fisher_information：该模型各参数在此数据集下的Fisher信息
    ################################################################################################
    def calcFisherInfo(self, model, data_loader, device="cpu"):
        fisher_information = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
        model.eval()
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            model.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()

            # 逐参数更新 Fisher 信息
            for name, param in model.named_parameters():
                fisher_information[name] += param.grad ** 2

        return fisher_information


    ################################################################################################
    # 功能：根据数据集大小加权平均Fisher信息
    # 输入：fisher_list：各数据集下的Fisher信息
    #       data_sizes：各数据集大小
    # 输出：avg_fisher：加权Fisher信息
    ################################################################################################
    def weightAvgFisher(self, fisher_list, data_sizes):
        if len(fisher_list) != len(data_sizes):
            raise ValueError("The length of fisher_list and data_sizes must be the same")

        total_size = sum(data_sizes)
        if total_size == 0:
            raise ValueError("Total size of data_sizes cannot be zero")

        avg_fisher = {}
        for name in fisher_list[0].keys():
            avg_fisher[name] = sum(
                fisher_info[name] * data_size for fisher_info, data_size in zip(fisher_list, data_sizes)) / total_size
        return avg_fisher


    ################################################################################################
    # 功能：根据加权Fisher信息，选择重要参数，创建二值掩码
    # 输入：fisher_information：各参数Fisher信息
    #       top_k_percent：选取的参数百分比数量
    # 输出：binary_mask：二值掩码，重要参数取1，非重要参数取0
    ################################################################################################
    def createBinaryMask(self, fisher_information, top_k_percent=5):
        # 将所有参数的 Fisher 信息平铺为一个列表
        importance_scores = []
        for name, fisher in fisher_information.items():
            importance_scores.extend(fisher.view(-1).tolist())

        # 排序并选择前 top_k_percent 的阈值
        threshold = sorted(importance_scores, reverse=True)[int(len(importance_scores) * top_k_percent / 100)]

        # 创建二值掩码
        binary_mask = {}
        for name, fisher in fisher_information.items():
            binary_mask[name] = (fisher >= threshold).float()  # 高于阈值的元素设置为 1，否则为 0

        return binary_mask


    ################################################################################################
    # 功能：评估当前恶意模型与良性模型的距离是否满足条件（论文中的公式11）
    # 输入：malicious_update：恶意模型更新
    #       benign_updates：良性模型更新
    # 输出：is_feasible：是否满足条件
    ################################################################################################
    def meetsDistanceCriteria(self, malicious_update, benign_updates):
        # 左式：恶意更新到每个良性更新的距离之和
        left_sum = 0
        for benign_update in benign_updates:
            distance = torch.sqrt(sum(torch.sum((malicious_update[name] - benign_update[name]) ** 2)
                                    for name in malicious_update))
            left_sum += distance

        # 右式：计算每个良性更新到其他良性更新的第4小距离并求和
        right_sum = 0
        for i, benign_update_i in enumerate(benign_updates):
            distances = []
            for j, benign_update_j in enumerate(benign_updates):
                if i != j:
                    dist = torch.sqrt(sum(torch.sum((benign_update_i[name] - benign_update_j[name]) ** 2)
                                        for name in benign_update_i))
                    distances.append(dist)

            distances.sort()
            fourth_smallest_distance = distances[3]  # 取第4小的距离
            right_sum += fourth_smallest_distance

        is_feasible = left_sum <= right_sum
        return is_feasible


    ################################################################################################
    # 功能：二分搜索最佳恶意提升系数delta
    # 输入：mean_update：良性更新均值
    #       std_update：良性更新方差
    #       binary_mask：重要参数二值掩码
    #       benign_updates：良性更新
    #       delta_max：最大恶意提升系数
    #       threshold：搜索精度
    #       max_iters：最大搜索次数
    # 输出：delta：最佳恶意提升系数
    ################################################################################################
    def binarySearchBoostingCoefficient(self, mean_update, std_update, binary_mask, benign_updates, delta_max = 10000, threshold=1, max_iters=30):
        low, high = 0, delta_max
        delta = delta_max / 2
        for _ in range(max_iters):
            candidate_update = {
                name: mean_update[name] - delta * binary_mask.get(name, torch.zeros_like(mean_update[name])) * std_update.get(name, torch.zeros_like(mean_update[name]))
                for name in mean_update
            }

            # 判断距离
            benign_updates_param = []
            for update in benign_updates:
                param, _ = self.calcMeanAndStd([update])
                benign_updates_param.append(param)
            if self.meetsDistanceCriteria(candidate_update, benign_updates_param):
                low = delta
            else:
                high = delta
            delta = (high + low) / 2
            if abs(high - low) < threshold:
                break
        return delta


    ################################################################################################
    # 功能：计算恶意更新
    # 输入：mean_update：良性更新均值
    #       std_update：良性更新方差
    #       binary_mask：重要参数二值掩码
    #       boosting_coefficient：恶意提升系数
    # 输出：malicious_update：恶意更新
    ################################################################################################
    def genMaliciousUpdate(self, mean_update, std_update, binary_mask, boosting_coefficient):
        malicious_update = {
            name: mean_update[name] - boosting_coefficient * binary_mask.get(name, torch.zeros_like(mean_update[name])) * std_update.get(name, torch.zeros_like(mean_update[name]))
            for name in mean_update
        }
        return malicious_update

    def create_model_from_update(self, model, update):
        with torch.no_grad():  # 确保不会跟踪梯度
            for name, param in model.named_parameters():
                if name in update:
                    # 确保参数尺寸匹配
                    param.copy_(update[name])
        return model


####################################################################################################
####################################################################################################