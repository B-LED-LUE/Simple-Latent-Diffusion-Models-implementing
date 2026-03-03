def train(model, loader, epochs, device=device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
    print("Training Start...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x0, _ in loader:
            x0 = x0.to(device)
            out, z = model(x0)
            loss = F.mse_loss(out, x0.view(-1, 28*28))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

model = AE()
epochs = 100
train(model, train_loader, epochs, device=device)
visual_triple_check(model, train_loader,16)