import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(x)


def train_fedadam(model, train_loader, global_model, mu, epochs=5, lr=0.01, test_loader=None, output_file=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    accuracy_list = []

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)

            # FedProx Regularization term
            prox_term = 0.5 * mu * sum(
                (p1 - p2).norm() for p1, p2 in zip(model.parameters(), global_model.parameters()))
            loss += prox_term

            loss.backward()
            optimizer.step()

        if test_loader:
            acc = evaluate_model(model, test_loader)
            accuracy_list.append(acc)
            print(f'Epoch {epoch + 1}/{epochs}, Accuracy: {acc:.2f}%')

    # 保存到文件
    if output_file:
        with open(output_file, 'w') as f:
            for epoch, acc in enumerate(accuracy_list):
                f.write(f'Epoch {epoch + 1}, Accuracy: {acc:.2f}%\n')

    return accuracy_list


def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy