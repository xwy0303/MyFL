import torch
import torch.nn as nn
import torch.optim as optim

# 定义经典的 CNN 神经网络模型
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        # 卷积层 1
        self.conv1 = nn.Conv2d(1, 16, 5)
        # 卷积层 2
        self.conv2 = nn.Conv2d(16, 32, 5)
        # 全连接层 1
        self.fc1 = nn.Linear(4 * 4 * 32, 120)
        # 全连接层 2
        self.fc2 = nn.Linear(120, 84)
        # 输出层
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积层 1 后接 ReLU 激活函数和最大池化
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        # 卷积层 2 后接 ReLU 激活函数和最大池化
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
        # 将特征图展平
        x = x.view(-1, 4 * 4 * 32)
        # 全连接层 1 后接 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc1(x))
        # 全连接层 2 后接 ReLU 激活函数
        x = torch.nn.functional.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        output = torch.nn.functional.log_softmax(x, dim=1)
        return output

def local_train(model, train_loader, optimizer, epoch, userid, train_losses, train_accuracies):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        if batch_idx % 100 == 0:
            train_losses.append(loss.item())
            train_accuracies.append(100. * correct / total)
    print('Train Epoch: {} userid: {}\tLoss: {:.6f}'.format(epoch, userid, loss.item()))