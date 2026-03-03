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
