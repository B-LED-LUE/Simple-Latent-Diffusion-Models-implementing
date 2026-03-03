import matplotlib.pyplot as plt

@torch.no_grad()
def generate_samples(VAE, num_samples=16):
    VAE.eval()
    z = torch.randn(num_samples, 16).to(device)
    samples = VAE.decode(z).cpu()


    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(samples[i].view(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.show()

generate_samples(VAE)