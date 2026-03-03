I have implemented a Simple Latent Diffusion Model (LDM) framework.

This project focuses on the core mechanism of how LDMs operate within the latent space. 
In this section, I won't dive into the heavy details—like specific conditioning (guiding the model) or the internal architecture of the U-Net. 
Instead, the focus is on the underlying mechanism that makes LDM so efficient.

For the diffusion process, I implemented both DDPM and DDIM. 
You can explore the theoretical deep dives and the "Delivery Man" narrative in my [Medium Post (Link)].
In this repository, you can explore the qualitative performance gap between a standard VAE and the Latent Diffusion Model (LDM). 

I provided a comprehensive comparison through three distinct lenses:
The Triple Check (AE): A direct comparison between the Original image, its Reconstruction, and a Randomly sampled generation.
VAE vs. LDM: A head-to-head battle showing why LDM consistently produces sharper, more valid samples than the "blurry" VAE.
Sampling Strategy (DDPM & DDIM):
DDPM/DDIM:[https://medium.com/@woohyun301/ddpm-ddim-a598d73d3216]
DDPM (Steps 1 & 2): Observing the probabilistic carving process.
DDIM: Achieving high-fidelity results with significantly fewer steps.


### 📊 Experimental Results

| Comparison Type | Visualization |
| :--- | :--- |
| **VAE vs. LDM** | <img src="https://github.com/user-attachments/assets/7f578d47-c8a2-447b-9903-232d44dfc0aa" width="100%"> |

> **Observation:** As seen in the results, the LDM (DDPM/DDIM) consistently "delivers" the latent variable $z$ to high-density regions. This results in significantly sharper samples compared to the standard VAE's "blurry" output, proving that our "Delivery Man" found the right address.
