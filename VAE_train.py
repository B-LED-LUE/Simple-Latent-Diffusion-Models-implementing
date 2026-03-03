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

