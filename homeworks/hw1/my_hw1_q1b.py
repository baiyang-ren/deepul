import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as data
import matplotlib.pyplot as plt

class DiscretizedLogisticMixture(nn.Module):
    def __init__(self, d, num_components=4):
        super().__init__()
        self.K = num_components  # number of logistic components
        self.d = d  # number of bins (e.g., 100 for histogram)
        # Each component has a mean, scale, and mixture weight
        self.means = nn.Parameter(torch.zeros(self.K), requires_grad=True)             # μ_k
        self.log_scales = nn.Parameter(torch.arange(self.K).float() / (self.K - 1) * d, requires_grad=True)        # log(s_k)
        self.logits = nn.Parameter(torch.randn(self.K), requires_grad=True)            # unnormalized mixture weights (π_k)

    def log_prob(self, x):  # x is a batch of integer bin indices
        """
        Compute log probability of each x under the mixture.
        x: LongTensor of shape (batch,)
        """
        x = x.float().unsqueeze(1).repeat(1,self.K)  # shape: (batch, 1)
        means = self.means.view(1, self.K)      # (1, K)
        scales = F.softplus(self.log_scales).view(1, self.K)  # (1, K)
        log_weights = F.log_softmax(self.logits, dim=0).view(1, self.K)  # (1, K)

        # Discretized CDF difference
        lower = x - 0.5
        upper = x + 0.5

        # Handle edge cases
        lower = torch.where(x == 0, torch.full_like(lower, -1e6), lower)
        upper = torch.where(x == 99, torch.full_like(upper, 1e6), upper)

        cdf_upper = torch.sigmoid((upper - means) / scales)
        cdf_lower = torch.sigmoid((lower - means) / scales)

        probs = cdf_upper - cdf_lower  # (batch, K)
        probs = torch.clamp(probs, min=1e-12)   # avoid log(0)
        log_probs = torch.log(probs) + log_weights  # (batch, K)

        return torch.logsumexp(log_probs, dim=1)  # total log-prob for each x

    def loss(self, x):
        return -self.log_prob(x).mean()  # average NLL over batch

    def get_distribution(self, num_bins=100):
        with torch.no_grad():
            device = self.means.device  # get model's device
            xs = torch.arange(num_bins, device=device).long()
            probs = torch.exp(self.log_prob(xs))
            return probs / probs.sum()  # normalize to sum to 1

def train_epochs(model, train_data, test_data, batch_size=256, epochs=10, lr=1e-2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_tensor = torch.tensor(train_data, dtype=torch.long)
    test_tensor = torch.tensor(test_data, dtype=torch.long)

    train_loader = data.DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_tensor, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss = model.loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Evaluate test loss
        model.eval()
        with torch.no_grad():
            total_loss = 0.0
            count = 0
            for batch in test_loader:
                batch = batch.to(device)
                loss = model.loss(batch)
                total_loss += loss.item() * batch.size(0)
                count += batch.size(0)
            test_loss = total_loss / count
            test_losses.append(test_loss)
            print(f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}")

    return np.array(train_losses), np.array(test_losses), model.get_distribution().cpu().numpy()

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
    return data[:split], data[split:]

train_data, test_data = q1_sample_data_2()
model = DiscretizedLogisticMixture(d=100, num_components=4)
train_losses, test_losses, model_probs = train_epochs(model, train_data, test_data, batch_size=128, epochs=100, lr=1e-1)

# Plot model vs empirical
empirical_counts = np.bincount(train_data, minlength=100)
empirical_probs = empirical_counts / empirical_counts.sum()

plt.figure(figsize=(10, 5))
plt.plot(empirical_probs, label='Empirical', drawstyle='steps-mid')
plt.plot(model_probs, label='Fitted (mixture)', linestyle='--')
plt.xlabel("Bin Index")
plt.ylabel("Probability")
plt.title("Model vs Empirical Histogram (Mixture of Logistics)")
plt.legend()
plt.grid(True)
plt.show()
