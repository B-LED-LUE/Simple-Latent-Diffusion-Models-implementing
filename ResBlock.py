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