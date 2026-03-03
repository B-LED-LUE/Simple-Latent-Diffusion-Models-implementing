import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

device = "cuda" if torch.cuda.is_available() else "cpu"
setting = transforms.Compose([
    transforms.ToTensor(),
])

all_data = datasets.MNIST(root='./data', train=True, download=True, transform=setting)

five_idx = (all_data.targets == 5).nonzero(as_tuple=True)[0]
train_loader = DataLoader(Subset(all_data, five_idx), batch_size=128, shuffle=True)
train_loader_all = DataLoader(all_data, batch_size=128, shuffle=True)

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, 28*28))
        out = self.decoder(z)
        return out, z

def train(model, loader, epochs, device=device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    print("Training Start...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x0, _ in loader:
            x0 = x0.to(device)
            out, z = model(x0)
            loss = F.mse_loss(out, x0.view(-1, 28*28))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

model = AE()
epochs = 100
train(model, train_loader, epochs, device=device)

import matplotlib.pyplot as plt

def visual_triple_check(model, data_loader, z_dim=16):
    model.eval()
    with torch.no_grad():
        # 1. Original data
        x_real, _ = next(iter(data_loader))
        x_real = x_real.to(device).view(-1, 784)
        
        # 2. Recon (Original -> Encoder -> Decoder)
        out_recon, _ = model(x_real)
        
        # 3. Random Sampling (Random Z -> Decoder)
        z_random = torch.randn(5, z_dim).to(device)
        out_sampled = model.decoder(z_random)

    # --- Visualization ---
    plt.figure(figsize=(10, 6))
    for i in range(5):
        # Fisrt row: Original
        plt.subplot(3, 5, i + 1)
        plt.imshow(x_real[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Original", loc='left', fontsize=10)

        # Second row: Reconstruction
        plt.subplot(3, 5, i + 6)
        plt.imshow(out_recon[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Recon", loc='left', fontsize=10)

        # Third row: Random sampling.
        plt.subplot(3, 5, i + 11)
        plt.imshow(out_sampled[i].cpu().view(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0: plt.title("Random", loc='left', fontsize=10)

    plt.tight_layout()
    plt.show()

visual_triple_check(model, train_loader,16)