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