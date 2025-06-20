import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Step 1: Data generator
def q1_sample_data_2():
    count = 100000
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

# Step 2: Convert to histogram
def compute_histogram(data, num_bins):
    hist = np.bincount(data, minlength=num_bins)
    return torch.tensor(hist, dtype=torch.float32)

# Step 3: Fit softmax model
def fit_histogram_softmax(train_data, test_data, num_bins=100, epochs=300):
    train_counts = compute_histogram(train_data, num_bins)
    test_counts = compute_histogram(test_data, num_bins)
    print(f'size of train counts: {train_counts.size()}')
    logits = nn.Parameter(torch.randn(num_bins))
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
    test_log_likelihood = (test_counts * torch.log(final_probs + 1e-9)).sum()
    test_nll = -test_log_likelihood
    print(f"Test NLL: {test_nll.item():.4f}")

    return final_probs, test_nll, train_counts / train_counts.sum()

# Step 4: Plotting
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

# Run everything
train_data, test_data = q1_sample_data_2()

# d=100
# fig, (ax1, ax2) = plt.subplots(1, 2)
# ax1.set_title("Train Data")
# ax1.hist(train_data, bins=np.arange(d) - 0.5, density=True)
# ax1.set_xlabel("x")
# ax2.set_title("Test Data")
# ax2.hist(test_data, bins=np.arange(d) - 0.5, density=True)
# print(f"Dataset {2}")
# plt.show()

# plot histogram of train data

model_probs, test_nll, empirical_probs = fit_histogram_softmax(train_data, test_data)
plot_comparison(empirical_probs, model_probs)
