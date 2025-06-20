import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Data generator
def q1_sample_data_2():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.4 + 0.05 * rand.randn(count)
    b = 0.5 + 0.10 * rand.randn(count)
    c = 0.7 + 0.02 * rand.randn(count)
    mask = np.random.randint(0, 3, size=count)
    samples = np.clip(a * (mask == 0) + b * (mask == 1) + c * (mask == 2), 0.0, 1.0)

    data = np.digitize(samples, np.linspace(0.0, 1.0, 100))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data

# Step 2: Compute histogram
def compute_histogram(data, num_bins, device):
    hist = np.bincount(data, minlength=num_bins)
    return torch.tensor(hist, dtype=torch.float32, device=device)

# Step 3: Fit model
def fit_histogram_softmax(train_data, test_data, num_bins=100, epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_counts = compute_histogram(train_data, num_bins, device)
    test_counts = compute_histogram(test_data, num_bins, device)

    logits = nn.Parameter(torch.randn(num_bins, device=device))
    optimizer = optim.Adam([logits], lr=0.1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        probs = torch.softmax(logits, dim=0)
        loss = -(train_counts * torch.log(probs + 1e-9)).sum()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Train NLL: {loss.item():.4f}")

    final_probs = torch.softmax(logits, dim=0).detach()
    test_nll = -(test_counts * torch.log(final_probs + 1e-9)).sum()
    print(f"Test NLL: {test_nll.item():.4f}")

    return final_probs.cpu(), test_nll.item(), (train_counts / train_counts.sum()).cpu()

# Step 4: Plot
def plot_comparison(empirical_probs, model_probs):
    x = np.linspace(0, 1, len(empirical_probs))
    plt.figure(figsize=(10, 5))
    plt.plot(x, empirical_probs.numpy(), label='Empirical (train)', drawstyle='steps-mid')
    plt.plot(x, model_probs.numpy(), label='Fitted (softmax)', linestyle='--')
    plt.xlabel("Bin (scaled from 0 to 1)")
    plt.ylabel("Probability")
    plt.title("Model vs Empirical Histogram Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run
train_data, test_data = q1_sample_data_2()
model_probs, test_nll, empirical_probs = fit_histogram_softmax(train_data, test_data)
plot_comparison(empirical_probs, model_probs)
