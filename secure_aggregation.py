from secretsharing import PlaintextToHexSecretSharer as SSP
import numpy as np

def secure_aggregation(models):
    secret_shares = []
    for model in models:
        parameters = []
        for param in model.parameters():
            parameters.append(param.detach().numpy().flatten())
        parameters = np.concatenate(parameters)

        secret = str(np.sum(parameters))
        shares = SSP.split_secret(secret, 2, len(models))
        secret_shares.append(shares)

    return secret_shares

if __name__ == "__main__":
    import torch
    from fedadam import SimpleNN

    node_models = [SimpleNN() for _ in range(5)]
    shares = secure_aggregation(node_models)
    for share in shares:
        print(share)