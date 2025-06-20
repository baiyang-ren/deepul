import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

# Step 1: Data generator
def q1_sample_data_2():
    count = 20000
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

# Step 2: Mini-batch training with logging
def fit_histogram_softmax_minibatch(train_data, test_data, num_bins=100, epochs=300, batch_size=20000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors and wrap in DataLoader
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    test_tensor = torch.tensor(test_data, dtype=torch.long)

    train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    # Learnable logits
    logits = nn.Parameter(torch.randn(num_bins, device=device))
    optimizer = optim.Adam([logits], lr=0.1)

    train_losses = []  # per mini-batch
    test_losses = []   # per epoch

    for epoch in range(epochs):
        for batch in train_loader:
            batch = batch.to(device)
            probs = torch.softmax(logits, dim=0)
            log_probs = torch.log(probs + 1e-9)
            loss = -log_probs[batch].sum() / batch.size(0)  # average loss per sample

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

    # Evaluate test loss (no need for logits.eval())
        with torch.no_grad():
            test_loss = 0.0
            total_samples = 0
            probs = torch.softmax(logits, dim=0)
            log_probs = torch.log(probs + 1e-9)
            for batch in test_loader:
                batch = batch.to(device)
                test_loss += -log_probs[batch].sum().item()
                total_samples += batch.size(0)
            avg_test_loss = test_loss / total_samples
            test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}, Avg Test Loss: {avg_test_loss:.4f}")

    final_probs = torch.softmax(logits, dim=0).detach().cpu()
    return final_probs, np.array(train_losses), np.array(test_losses)

# Step 3: Plotting
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

# Step 4: Run training
train_data, test_data = q1_sample_data_2()
model_probs, train_losses, test_losses = fit_histogram_softmax_minibatch(train_data, test_data)

# Optional: Empirical distribution for comparison
empirical_counts = np.bincount(train_data, minlength=100)
empirical_probs = torch.tensor(empirical_counts / empirical_counts.sum(), dtype=torch.float32)

# Plot distribution
plot_comparison(empirical_probs, model_probs)

# Optional: Plot losses
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss (per batch)", alpha=0.6)
plt.xlabel("Mini-batch Iteration")
plt.ylabel("Loss")
plt.title("Training Loss per Mini-batch")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(test_losses, marker='o', label="Test Loss (per epoch)")
plt.xlabel("Epoch")
plt.ylabel("Avg Test Loss")
plt.title("Test Loss per Epoch")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
