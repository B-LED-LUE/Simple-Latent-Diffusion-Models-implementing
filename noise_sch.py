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