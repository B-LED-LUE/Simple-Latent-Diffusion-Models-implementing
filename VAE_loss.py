def loss_function( rebuilt_x, x, mu, logvar,device = device):
  MSE = F.mse_loss(rebuilt_x, x.view(-1,784), reduction = 'sum')
  KLD = -0.5 * torch.sum(logvar + 1 - logvar.exp() - mu.pow(2))
  return MSE, KLD