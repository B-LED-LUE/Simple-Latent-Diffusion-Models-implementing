import matplotlib.pyplot as plt

@torch.no_grad()
def generate_samples(VAE, LDM,noise_schedule,num_samples=16):
    VAE.eval()
    z = torch.randn(num_samples, 16).to(device)
    ddpm_1 = reverse_process_DDPM(LDM, noise_schedule, device, num_samples)
    ddpm_2 = reverse_process_DDPM_aprox(LDM, noise_schedule, device, num_samples)
    DDIM = reverse_process_DDIM(LDM, noise_schedule, device, num_samples, steps = 50, eta = 0.7)

    samples = VAE.decode(z).cpu()
    samples2 = VAE.decode(ddpm_1).cpu()
    samples3 = VAE.decode(ddpm_2).cpu()
    samples4 = VAE.decode(DDIM).cpu()

    print("random_z")
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(samples[i].view(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.show()

    print("beta_tilde")
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(samples2[i].view(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.show()

    print("beta_aprox")
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(samples3[i].view(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.show()
    
    print("DDIM")
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples, 1))
    for i in range(num_samples):
        axes[i].imshow(samples4[i].view(28, 28), cmap='gray')
        axes[i].axis('off')
    plt.show()

generate_samples(VAE, LDM, noise_schedule)