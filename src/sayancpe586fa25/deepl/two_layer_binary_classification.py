import torch

def binary_classification(d, n, epochs=10000, eta=0.001):
    """
    Performs binary classification using gradient descent and autograd.

    Args:
        d (int): number of features
        n (int): number of samples
        epochs (int): number of training epochs (default 10000)
        eta (float): learning rate (default 0.001)

    Returns:
        W1, W2, W3, W4: trained weight matrices
        loss_history: list containing loss value at each epoch
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate feature matrix
    X = torch.randn(n, d, dtype=torch.float32, device=device)

    # Generate labels
    Y = (X.sum(dim=1, keepdim=True) > 2).float()

    # Weight initialization (Gaussian, He-style)
    W1 = torch.randn(d, 48, device=device) * (2 / d) ** 0.5
    W2 = torch.randn(48, 16, device=device) * (2 / 48) ** 0.5
    W3 = torch.randn(16, 32, device=device) * (2 / 16) ** 0.5
    W4 = torch.randn(32, 1, device=device) * (2 / 32) ** 0.5

    for W in [W1, W2, W3, W4]:
        W.requires_grad_()

    loss_history = []

    # Training loop
    for _ in range(epochs):
        A1 = torch.sigmoid(X @ W1)
        A2 = torch.sigmoid(A1 @ W2)
        A3 = torch.sigmoid(A2 @ W3)
        Y_hat = torch.sigmoid(A3 @ W4)

        loss = -torch.mean(
            Y * torch.log(Y_hat + 1e-8) +
            (1 - Y) * torch.log(1 - Y_hat + 1e-8)
        )

        loss_history.append(loss.item())

        loss.backward()

        with torch.no_grad():
            for W in [W1, W2, W3, W4]:
                W -= eta * W.grad
                W.grad.zero_()

    return W1, W2, W3, W4, loss_history

