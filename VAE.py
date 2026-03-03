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