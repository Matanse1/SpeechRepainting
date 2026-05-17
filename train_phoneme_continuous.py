
def training_loss(
    net,
    loss_fn,
    melspec,
    masked_cond,
    mask,
    mask_mask,
    phoneme_target,
    phoneme_target_mask,
    diffusion_hyperparams,
    masked_audio_time_mask,
    on_noisy_masked_melspec,
    w_masked_pix=0.7,
    sde=None,
):
    if sde is None:
        _dh = diffusion_hyperparams
        T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]
        B, C, L = melspec.shape
        diffusion_steps = torch.randint(T, size=(B, 1, 1)).cuda()
        z = torch.normal(0, 1, size=melspec.shape).cuda()
        if on_noisy_masked_melspec:
            transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
            transformed_X = melspec * torch.unsqueeze(mask, dim=1) + transformed_X * (1 - torch.unsqueeze(mask, dim=1))
        else:
            transformed_X = torch.sqrt(Alpha_bar[diffusion_steps]) * melspec + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
        phoneme_estimated = net(transformed_X, masked_cond, diffusion_steps.view(B, 1), cond_drop_prob=0, mask_padding=masked_audio_time_mask)
    else:
        B = melspec.shape[0]
        device = melspec.device
        eps = 1e-5
        t = torch.rand(B, device=device) * (sde.T - eps) + eps
        z = torch.randn_like(melspec)
        mean, std = sde.marginal_prob(melspec, None, t)
        std_expanded = std[:, None, None]
        x_t = mean + std_expanded * z
        if on_noisy_masked_melspec:
            x_t = melspec * torch.unsqueeze(mask, dim=1) + x_t * (1 - torch.unsqueeze(mask, dim=1))
        phoneme_estimated = net(x_t, masked_cond, t.view(B, 1), cond_drop_prob=0, mask_padding=masked_audio_time_mask)

    loss = loss_fn(phoneme_estimated, phoneme_target)
    loss = loss * phoneme_target_mask
    unmaksed_loss = torch.sum(mask * loss) / torch.sum(mask * mask_mask)
    masked_loss = torch.sum((1 - mask) * loss) / torch.sum((1 - mask) * mask_mask)
    return (1 - w_masked_pix) * unmaksed_loss + w_masked_pix * masked_loss





def test_loss(*args, **kwargs):
    return training_loss(*args, **kwargs)