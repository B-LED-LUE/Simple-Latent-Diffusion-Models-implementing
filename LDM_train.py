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
