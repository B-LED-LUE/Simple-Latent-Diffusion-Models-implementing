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

class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()

    self.fc1 = nn.Linear(28*28, 400)
    self.fc2_mu = nn.Linear(400, 20)
    self.fc2_logvar = nn.Linear(400, 20)

    self.fc3 = nn.Linear(20, 400)
    self.fc4 = nn.Linear(400, 28*28)

  def encode(self, x):
    h1 = F.relu(self.fc1(x))
    return self.fc2_mu(h1), self.fc2_logvar(h1)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    h3 = F.relu(self.fc3(z))
    return torch.sigmoid(self.fc4(h3))

  def forward(self, x):
    mu, logvar = self.encode(x.view(-1, 28*28))
    z = self.reparameterize(mu, logvar)
    return self.decode(z), mu, logvar

def loss_function( rebuilt_x, x, mu, logvar,device = device):
  MSE = F.mse_loss(rebuilt_x, x.view(-1,784), reduction = 'sum')
  KLD = -0.5 * torch.sum(logvar + 1 - logvar.exp() - mu.pow(2))
  return MSE, KLD

#-------------------------------------------------------------------- train
VAE = VAE_model().to(device)
optimizer = torch.optim.AdamW(VAE.parameters(), lr=5e-4)

def train(epoch, current_step, total_steps, device):
  VAE.train()
  total_loss = 0
  for data, _ in train_loader:
    data = data.to(device)
    optimizer.zero_grad()

    beta = min(1.0, current_step / (total_steps * 0.5))
    rebuilt_batch, mu, logvar = VAE(data)

    cross_loss, KLD_loss = loss_function(rebuilt_batch, data, mu, logvar, device)

    loss = cross_loss + beta * KLD_loss
    
    loss.backward()

    total_loss += loss.item()

    optimizer.step()

    current_step += 1
  print(f"Epoch {epoch}: Avg Loss {total_loss / len(train_loader.dataset):.4f}")
  return current_step

epochs = 1000
total_steps = len(train_loader) * epochs
current_step = 0

for epoch in range(epochs):
  current_step = train(epoch, current_step, total_steps, device)

#-----------------------------------------------------------------Generate
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

#------------------------------------------------------------------LDM
class TimeEmbedding(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    self.mlp = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.SiLU(),
        nn.Linear(dim *4, dim)
    )
  def forward(self, t):
    device = t.device
    half_dim = self.dim // 2

    x = math.log(10000)/(half_dim - 1)
    x = torch.exp(torch.arange(half_dim, device = device)* - x)
    x = t[:,None]*x[None,:]
    x = torch.cat((x.sin(), x.cos()), dim = -1)
    x = self.mlp(x)
    return x

class ResBlock(nn.Module):
  def __init__(self, in_ch, out_ch, time_dim):
    super().__init__()

    self.time_mlp = nn.Sequential(
        nn.SiLU(),
        nn.Linear(time_dim, out_ch)
    )

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding = 1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_ch, out_ch, 3, padding = 1),
        nn.GroupNorm(8, out_ch),
        nn.SiLU()
    )

    self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch !=  out_ch else nn.Identity()

  def forward(self, x, t_emb):
    h = self.conv1(x)
    t_emb_refined = self.time_mlp(t_emb).view(-1, h.shape[1],1,1)
    h = h + t_emb_refined
    h = self.conv2(h)
    return h + self.shortcut(x)

class Unet(nn.Module):
  def __init__(self, in_ch = 1, time_dim = 128):
    super().__init__()
    self.time_mlp = TimeEmbedding(time_dim)
    self.init_conv = nn.Conv2d(in_ch, 64, 3, padding = 1)

    self.down1 = ResBlock(64, 128, time_dim)
    self.pool1 = nn.Conv2d(128, 128, 3, stride = 2, padding = 1)

    self.down2 = ResBlock(128, 256, time_dim)
    self.pool2 = nn.Conv2d(256, 256, 3, stride = 2, padding = 1)

    self.mid1 = ResBlock(256, 512, time_dim)
    self.mid2 = ResBlock(512, 256, time_dim)

    self.up1 = nn.Upsample(scale_factor = 2, mode = 'bilinear')
    self.up_res1 = ResBlock(256+256, 128, time_dim)
    self.up2 = nn.Upsample(scale_factor = 2, mode = 'bilinear')
    self.up_res2 = ResBlock(128 + 128, 64, time_dim)

    self.out_conv = nn.Conv2d(64, in_ch, 1)

  def forward(self, x, t):
    t_emb = self.time_mlp(t)
    x = x.view(-1,1,4,4)
    x1 = self.init_conv(x)
    x2 = self.down1(x1, t_emb)
    x3 = self.pool1(x2)
    x4 = self.down2(x3, t_emb)
    x5 = self.pool2(x4)

    x6 = self.mid1(x5, t_emb)
    x7 = self.mid2(x6, t_emb)

    x = self.up1(x7)
    if x4.dim() == 2:
      x4 = x4.view(x4.shape[0], -1, x.size(2), x.size(3))

    if x.shape[2:] != x4.shape[2:]:
      x = F.interpolate(x, size=x4.shape[2:], mode='bilinear', align_corners=False)
    x = torch.cat([x,x4], dim=1)
    x = self.up_res1(x, t_emb)
    x = self.up2(x)
    if x2.dim() == 2:
      x2 = x2.view(x2.shape[0], -1, x.size(2), x.size(3))

    if x.shape[2:] != x4.shape[2:]:
      x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
    x = torch.cat([x,x2], dim = 1)
    x = self.up_res2(x, t_emb)
    out =  self.out_conv(x)
    return out.view(out.size(0), -1)

class noise_scheduling(nn.Module):
  def __init__(self, T = 1000, beta_start=1e-4, beta_end = 0.02, device = device):
    super().__init__()
    self.T = T
    self.betas = torch.linspace(beta_start, beta_end, T, device = device)
    self.alphas = 1 - self.betas
    self.alphas_bars = torch.cumprod(self.alphas, dim = 0)
    self.sqrt_alphas_bars = torch.sqrt(self.alphas_bars)
    self.sqrt_one_minus_alphas_bars = torch.sqrt(1-self.alphas_bars)

  def forward(self, z0, t):
    noise = torch.randn_like(z0)
    sqrt_alphas_bars = self.sqrt_alphas_bars[t].view(-1,1)
    sqrt_one_minus_alphas_bars = self.sqrt_one_minus_alphas_bars[t].view(-1,1)
    return sqrt_alphas_bars * z0 + sqrt_one_minus_alphas_bars * noise, noise

def forward_process(LDM, VAE, data, noise_schedule, epochs, device =device):
  VAE.to(device)
  VAE.eval()
  LDM.to(device)
  optimizer = torch.optim.AdamW(LDM.parameters(), lr = 5e-4)
  print("start")
  for epoch in range(epochs):
    LDM.train()
    epoch_loss = 0

    for img, _ in data:
      img = img.to(device)
      img = img.view(img.size(0), -1)
      with torch.no_grad():
        mu, logvar = VAE.encode(img)
        z0 = VAE.reparameterize(mu, logvar)
      t = torch.randint(0, noise_schedule.T, (mu.shape[0],), device = device)
      zt, noise_t = noise_schedule(z0, t)

      pred = LDM(zt,t)
      loss = F.mse_loss(pred, noise_t)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      epoch_loss += loss.item()
    print(f"Epoch[{epoch+1}/{epochs}]| Loss:{loss.item():.6f}")

LDM = Unet(in_ch = 1, time_dim = 128).to(device)
noise_schedule = noise_scheduling(T = 1000, beta_start = 1e-4, beta_end = 0.02,device = device)
data = train_loader

forward_process(LDM, VAE, data, noise_schedule, epochs = 30, device = device)




@torch.no_grad()
def reverse_process_DDPM(LDM, noise_schedule, device, n):
  LDM.eval()
  zt = torch.randn(n, 16, device = device)
  for t in reversed(range(noise_schedule.T)):
    t_batch = torch.full((n,), t, device = device).long()
    pred = LDM(zt, t_batch)
    alpha = noise_schedule.alphas[t]
    alpha_bar_t = noise_schedule.alphas_bars[t]

    beta = noise_schedule.betas[t]
    beta_tilde = (1-noise_schedule.alphas_bars[t-1])/(1-alpha_bar_t)*beta if t > 0 else 0

    eps = torch.randn_like(zt) if t > 0 else 0
    zt = ((1 / torch.sqrt(alpha))*(zt-beta/(torch.sqrt(1-alpha_bar_t))*pred)
    +math.sqrt(beta_tilde)*eps)
  return zt

@torch.no_grad()
def reverse_process_DDPM_aprox(LDM, noise_schedule, device, n):
  LDM.eval()
  zt = torch.randn(n, 16, device = device)
  for t in reversed(range(noise_schedule.T)):
    t_batch = torch.full((n,), t, device = device).long()
    pred = LDM(zt, t_batch)
    alpha = noise_schedule.alphas[t]
    alpha_bar_t = noise_schedule.alphas_bars[t]

    beta = noise_schedule.betas[t]

    eps = torch.randn_like(zt) if t > 0 else 0
    zt = ((1 / torch.sqrt(alpha))*(zt-beta/(torch.sqrt(1-alpha_bar_t))*pred)
    +torch.sqrt(beta)*eps)

  return zt

@torch.no_grad()
def reverse_process_DDIM(LDM, noise_schedule, device,n,steps = 50, eta = 0.7):
  LDM.eval()
  steps = steps
  batch_size = n
  zt = torch.randn(batch_size, 16, device = device)
  t = torch.linspace(noise_schedule.T-1, 0, steps, device = device).long()
  t_section = list(zip(t[:-1],t[1:]))

  for t_start, t_end in t_section:
    eps = torch.randn_like(zt, device = device)
    t_batch = torch.full((batch_size,), t_start, device = device).long()
    pred_eps = LDM(zt, t_batch)
    alpha_bar_t = noise_schedule.alphas_bars[t_start]
    alpha_bar_t_minus_one = noise_schedule.alphas_bars[t_end]
    pred_z0 = (zt - torch.sqrt(1-alpha_bar_t) * pred_eps) / torch.sqrt(alpha_bar_t)
    sigma_t = eta * ((1-alpha_bar_t_minus_one)/(1-alpha_bar_t))*((alpha_bar_t_minus_one - alpha_bar_t) / alpha_bar_t_minus_one)

    zt = (torch.sqrt(alpha_bar_t_minus_one)*pred_z0
          + torch.sqrt(1-alpha_bar_t_minus_one - sigma_t) * pred_eps
          + torch.sqrt(sigma_t)*eps)
  return zt

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