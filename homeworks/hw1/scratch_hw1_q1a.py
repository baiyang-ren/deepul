import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
import torch.optim as optim

from deepul.utils import (
    get_data_dir,
    load_colored_mnist_text,
    load_pickled_data,
    load_text_data,
    save_distribution_1d,
    save_distribution_2d,
    save_text_to_plot,
    save_timing_plot,
    save_training_plot,
    savefig,
    show_samples,
)

quiet = False

def q1_sample_data_1(d=20):
    count = 1000
    rand = np.random.RandomState(0)
    samples = 0.2 + 0.2 * rand.randn(count)
    data = np.digitize(samples, np.linspace(0.0, 1.0, d))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data, d

def q1_sample_data_2(d=100):
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.4 + 0.05 * rand.randn(count)
    b = 0.5 + 0.10 * rand.randn(count)
    c = 0.7 + 0.02 * rand.randn(count)
    mask = np.random.randint(0, 3, size=count)
    samples = np.clip(a * (mask == 0) + b * (mask == 1) + c * (mask == 2), 0.0, 1.0)
    data = np.digitize(samples, np.linspace(0.0, 1.0, d))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data, d

def train_step(model, train_loader, optimizer):
    model.train()
    train_losses = []
    for x in train_loader:
        x = x.to(model.logits.device)
        loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def eval_step(model, test_loader):
    model.eval()
    total_loss=0.0
    with torch.no_grad():
        for x in test_loader:
            x = x.to(model.logits.device)
            loss = model.loss(x)
            total_loss = total_loss+loss*x.shape[0]
        avg_loss = total_loss/len(test_loader.dataset)
    return avg_loss.item()

def train_epochs(model, train_loader, test_loader, batch_size=128, epochs=100, lr=1e-1):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    test_losses = []
    # train loop iterate epochs
    for epoch in range(epochs):
        train_loss_per_epoch = train_step(model, train_loader, optimizer)
        train_losses.extend(train_loss_per_epoch)
        test_loss_per_epoch = eval_step(model, test_loader)
        test_losses.append(test_loss_per_epoch)
        if not quiet:
            print(f'Epoch {epoch}, Test loss {test_loss_per_epoch:.4f}')
    return train_losses, test_losses
# Question 1

# Define a maximum likelihood model with softmax
class SoftmaxModel(nn.Module):
    def __init__(self, d):
        super(SoftmaxModel, self).__init__()
        self.logits = nn.Parameter(torch.zeros(d))  # unnormalized logits for each bin

    def forward(self, x):
        return F.softmax(self.logits, dim=0)
    
    def loss(self, x):
        """
        Compute the negative log likelihood loss for the given data.
        x: LongTensor of shape (batch_size,)
        """
        probs = self.forward(x)
        logits = self.logits.unsqueeze(0).repeat(x.shape[0], 1)  # shape: (batch_size, d)
        # return -torch.sum(torch.log(probs[x]) + 1e-9)
        return F.cross_entropy(logits, x, reduction='mean')

    def get_distribution(self):
        distribution = F.softmax(self.logits, dim=0)
        return distribution.detach().cpu().numpy()



# Sample data
# train_data, test_data, d= q1_sample_data_1(d=20)
train_data, test_data, d= q1_sample_data_2(d=100)

# Data loaders for minibatch training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = SoftmaxModel(d).to(device)
# model = DiscretizedLogisticMixture(d=d, n_mix=4).to(device)
train_loader = data.DataLoader(train_data, batch_size=256, shuffle=True)
test_loader = data.DataLoader(test_data, batch_size=256)    
train_losses, test_losses = train_epochs(model, train_loader, test_loader,epochs=20, lr=1e-2)
distribution = model.get_distribution()

save_training_plot(
    train_losses,
    test_losses,
    f"Q1({0}) Dataset {0} Train Plot",
    f"results/q1_{0}_dset{0}_train_plot.png",
)
save_distribution_1d(
    train_data,
    distribution,
    f"Q1({0}) Dataset {0} Learned Distribution",
    f"results/q1_{0}_dset{0}_learned_dist.png",
)
# # plot the histogram of the training data and test data in one figure
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.hist(train_data, bins=np.arange(d) - 0.5, density=True, alpha=0.5, label='Train Data')
# plt.xlabel('Bin index')
# plt.ylabel('Density')
# plt.title('Histogram of Training Data')
# plt.legend()        
# plt.subplot(1, 2, 2)
# plt.hist(test_data, bins=np.arange(d) - 0.5, density=True, alpha=0.5, label='Test Data')
# plt.xlabel('Bin index')
# plt.ylabel('Density')
# plt.title('Histogram of Test Data')     
# plt.legend()
# plt.tight_layout()
# plt.show()