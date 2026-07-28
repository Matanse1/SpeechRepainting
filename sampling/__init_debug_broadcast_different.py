"""Debug PC sampler that preserves the input mel tensor dimensionality.

This module is intentionally separate from :mod:`sampling.__init__`.  It is
used to test whether hard-coded four-dimensional broadcasting in the normal
predictor/corrector path affects inference whose state is ``[B, F, T]``.
"""

import numpy as np
import torch

from .correctors import CorrectorRegistry
from .predictors import PredictorRegistry


def _batch_coefficient(value, reference):
    """Reshape a scalar/batch coefficient for broadcasting over ``reference``."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=reference.device, dtype=reference.dtype)
    else:
        value = value.to(device=reference.device, dtype=reference.dtype)

    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    elif value.numel() == 1 and reference.shape[0] != 1:
        value = value.reshape(1).expand(reference.shape[0])
    else:
        value = value.reshape(reference.shape[0])

    return value.reshape(reference.shape[0], *([1] * (reference.ndim - 1)))


def get_pc_sampler(
    predictor_name,
    corrector_name,
    sde,
    score_fn,
    y,
    denoise=True,
    eps=3e-2,
    snr=0.1,
    corrector_steps=1,
    probability_flow=False,
    intermediate=False,
    w_mel_cond=0.0,
    mask=None,
    mask_noise=False,
    debug_shapes=True,
    **kwargs,
):
    """Create a PC sampler without changing ``y.ndim`` during sampling.

    The public signature matches the project's regular ``get_pc_sampler`` so
    the inference call can be switched between implementations for comparison.
    Supported update rules are the predictor/corrector names already registered
    by the project.  The actual update equations are reproduced here only to
    replace fixed ``[:, None, None, None]`` broadcasting with broadcasting based
    on the current state dimensionality.
    """
    del intermediate, w_mel_cond, kwargs

    # Validate names using the same registries as the regular sampler.
    PredictorRegistry.get_by_name(predictor_name)
    CorrectorRegistry.get_by_name(corrector_name)

    rsde = sde.reverse(score_fn, probability_flow=probability_flow)
    initial_shape = tuple(y.shape)

    def assert_state_shape(name, value):
        if tuple(value.shape) != initial_shape:
            raise RuntimeError(
                f"{name} changed PC state shape from {initial_shape} "
                f"to {tuple(value.shape)}"
            )

    def apply_masking(x, t):
        if mask is None:
            return x

        current_mask = mask.to(device=x.device, dtype=x.dtype)
        observed = y.to(device=x.device, dtype=x.dtype)

        if tuple(current_mask.shape) != initial_shape:
            raise RuntimeError(
                f"mask shape {tuple(current_mask.shape)} does not match "
                f"state shape {initial_shape}"
            )
        if tuple(observed.shape) != initial_shape:
            raise RuntimeError(
                f"condition shape {tuple(observed.shape)} does not match "
                f"state shape {initial_shape}"
            )

        if mask_noise:
            mean, std = sde.marginal_prob(observed, None, t)
            observed = mean + _batch_coefficient(std, observed) * torch.randn_like(observed)

        # Project convention: mask == 1 is known; mask == 0 is missing.
        result = observed * current_mask + x * (1 - current_mask)
        assert_state_shape("masking", result)
        return result

    def corrector_update(x, t):
        if corrector_name == "none":
            return x, x

        x_mean = x
        for _ in range(corrector_steps):
            grad = score_fn(x, y, t)
            assert_state_shape("corrector score", grad)
            noise = torch.randn_like(x)

            if corrector_name == "langevin":
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (snr * noise_norm / (grad_norm + 1e-12)) ** 2 * 2
            elif corrector_name == "ald":
                std = sde.marginal_prob(x, y, t)[1]
                step_size = (snr * std) ** 2 * 2
            else:
                raise ValueError(f"Unsupported debug corrector: {corrector_name}")

            step = _batch_coefficient(step_size, x)
            x_mean = x + step * grad
            x = x_mean + torch.sqrt(step * 2) * noise
            assert_state_shape("corrector output", x)

        return x, x_mean

    def predictor_update(x, t, stepsize):
        if predictor_name == "none":
            return x, x

        noise = torch.randn_like(x)
        if predictor_name == "euler_maruyama":
            dt = -1.0 / rsde.N
            drift, diffusion = rsde.sde(x, y, t)
            x_mean = x + drift * dt
            diffusion = _batch_coefficient(diffusion, x)
            x = x_mean + diffusion * np.sqrt(-dt) * noise
        elif predictor_name == "reverse_diffusion":
            drift, diffusion = rsde.discretize(x, y, t, stepsize)
            x_mean = x - drift
            diffusion = _batch_coefficient(diffusion, x)
            x = x_mean + diffusion * noise
        else:
            raise ValueError(f"Unsupported debug predictor: {predictor_name}")

        assert_state_shape("predictor mean", x_mean)
        assert_state_shape("predictor output", x)
        return x, x_mean

    def pc_sampler():
        with torch.no_grad():
            xt = sde.prior_sampling(y.shape, y).to(y.device)
            assert_state_shape("prior sample", xt)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
            last_mean = xt

            if debug_shapes:
                print(
                    "[BROADCAST DEBUG] initial shapes:",
                    f"state={tuple(xt.shape)}",
                    f"condition={tuple(y.shape)}",
                    f"mask={None if mask is None else tuple(mask.shape)}",
                )

            for i, timestep in enumerate(timesteps):
                if i != len(timesteps) - 1:
                    stepsize = timestep - timesteps[i + 1]
                else:
                    stepsize = timesteps[-1]
                vec_t = torch.ones(y.shape[0], device=y.device) * timestep

                xt = apply_masking(xt, vec_t)
                xt, last_mean = corrector_update(xt, vec_t)
                xt = apply_masking(xt, vec_t)
                xt, last_mean = predictor_update(xt, vec_t, stepsize)
                xt = apply_masking(xt, vec_t)

                if debug_shapes and i == 0:
                    print(
                        "[BROADCAST DEBUG] after first PC step:",
                        f"state={tuple(xt.shape)}",
                        f"mean={tuple(last_mean.shape)}",
                    )

            result = last_mean if denoise else xt
            final_t = torch.ones(y.shape[0], device=y.device) * eps
            result = apply_masking(result, final_t)
            assert_state_shape("final result", result)

            if debug_shapes:
                print("[BROADCAST DEBUG] final shape:", tuple(result.shape))

            nfe = sde.N * (corrector_steps + 1)
            return result, nfe

    return pc_sampler

