import torch
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
