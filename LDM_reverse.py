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