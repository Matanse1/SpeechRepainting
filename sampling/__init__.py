# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
import torch

from .predictors import Predictor, PredictorRegistry
from .correctors import Corrector, CorrectorRegistry

__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_pc_sampler', 'get_ode_sampler', 'get_sb_sampler'
]

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
    probability_flow: bool = False,
    intermediate=False,
    w_mel_cond=0.0,
    # Masking / inpainting
    mask=None,
    mask_noise=False,
    **kwargs
):

    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        mask: Optional mask tensor with same shape as `y` (1 for observed/preserved values, 0 for unknown values).
        mask_noise: If True, the observed portion is replaced by a noisy version of `y` at every timestep (useful for inpainting).
        **kwargs: Additional arguments forwarded to `score_fn` and `guidance_fn`.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """

    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(
        sde,
        score_fn,
        snr=snr,
        n_steps=corrector_steps,
        mask=mask,
    )

    # def _apply_masking(x, y, t):
    #     """Apply masking / inpainting constraints to the current state."""
    #     if mask is None:
    #         return x

    #     if mask_noise:
    #         # For inpainting with noise injection, replace observed entries with a noisy version at time t.
    #         mean, std = sde.marginal_prob(y, y, t)
    #         noisy_y = mean + std[:, None, None] * torch.randn_like(y)
    #         return noisy_y * mask + x * (1 - mask)

    def _apply_masking(x, y, t):
        if mask is None:
            return x

        _mask = mask.to(x.device)
        _y = y.to(x.device)

        # Match shapes:
        # x may be [B, 1, 80, T], while y/mask may be [B, 80, T]
        if x.ndim == 4 and _y.ndim == 3:
            _y = _y.unsqueeze(1)
        if x.ndim == 3 and _y.ndim == 4 and _y.shape[1] == 1:
            _y = _y.squeeze(1)

        if x.ndim == 4 and _mask.ndim == 3:
            _mask = _mask.unsqueeze(1)
        if x.ndim == 3 and _mask.ndim == 4 and _mask.shape[1] == 1:
            _mask = _mask.squeeze(1)

        if mask_noise:
            z = torch.randn_like(_y)
            mean, std = sde.marginal_prob(_y, None, t)

            while std.ndim < _y.ndim:
                std = std.unsqueeze(-1)

            _y_t = mean + std * z
        else:
            _y_t = _y

        # mask == 1: known/unmasked region
        # mask == 0: missing/masked region
        return _y_t * _mask + x * (1 - _mask)

    def pc_sampler():
        """Predictor-corrector sampler with one projection per transition."""
        with torch.no_grad():
            batch_size = y.shape[0]
            device = y.device

            xt = sde.prior_sampling(y.shape, y).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            # Initialize the known region at time T.
            initial_t = torch.full((batch_size,), float(sde.T), device=device)

            xt = _apply_masking(xt, y, initial_t)

            for i, t in enumerate(timesteps):
                is_last_step = i == len(timesteps) - 1

                if is_last_step:
                    # Final transition: eps -> 0.
                    next_t = torch.zeros_like(t)
                else:
                    next_t = timesteps[i + 1]

                stepsize = t - next_t

                vec_t = torch.full((batch_size,), t, device=device)

                # xt was already projected at time t by the previous iteration.
                # The corrector is mask-aware and preserves the known region.
                xt, xt_mean = corrector.update_fn(xt, y, vec_t)

                # Predictor moves the state from t to next_t.
                xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)

                # Preserve the original denoising behavior on the final step.
                if denoise and is_last_step:
                    state_for_next_time = xt_mean
                else:
                    state_for_next_time = xt

                vec_next_t = torch.full(
                    (batch_size,),
                    next_t,
                    device=device,
                )

                # The only projection in this iteration.
                # This prepares the state for the next corrector timestep.
                xt = _apply_masking(
                    state_for_next_time,
                    y,
                    vec_next_t,
                )

            ns = sde.N * (corrector.n_steps + 1)
            return xt, ns

    return pc_sampler

    # def pc_sampler():
    #     """The PC sampler function."""
    #     with torch.no_grad():
    #         xt = sde.prior_sampling(y.shape, y).to(y.device)
    #         timesteps = torch.linspace(sde.T, eps, sde.N, device=y.device)
    #         for i in range(sde.N):
    #             t = timesteps[i]
    #             if i != len(timesteps) - 1:
    #                 stepsize = t - timesteps[i + 1]
    #             else:
    #                 stepsize = timesteps[-1]  # from eps to 0
    #             vec_t = torch.ones(y.shape[0], device=y.device) * t

    #             # Apply masking / inpainting constraints
    #             xt = _apply_masking(xt, y, vec_t)
                
    #             xt, xt_mean = corrector.update_fn(xt, y, vec_t)
    #             xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)

    #         x_result = xt_mean if denoise else xt
    #         ns = sde.N * (corrector.n_steps + 1)
    #         return x_result, ns

    # return pc_sampler


def get_ode_sampler(
    sde,
    score_fn,
    y,
    mask=None,
    on_noisy_masked_melspec=True,
    method="heun",
    steps=None,
    denoise=True,
    eps=3e-2,
):
    """Create a fixed-step probability-flow ODE sampler using Heun's method.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: Conditioning mel spectrogram and shape reference.
        mask: Optional inpainting mask (1 observed, 0 generated).
        on_noisy_masked_melspec: Match the training-time input convention. If
            ``True``, the observed part of ``x_t`` is kept clean. If ``False``,
            it follows its forward marginal using one fixed noise realization
            for the complete ODE trajectory.
        method: Must be ``"heun"``. It is explicit in the configuration so
            experiment logs describe the numerical solver being used.
        steps: Number of fixed ODE steps. Defaults to ``sde.N``.
        denoise: If ``True``, estimate the clean sample at the final time.
        eps: Small positive final time used for numerical stability.

    Returns:
        A function that returns ``(sample, number_of_score_evaluations)``.

    Notes:
        This is a deterministic ODE after the initial latent ``x_T`` and the
        optional fixed observed-region noise are chosen. Heun's
        predictor/corrector stages are numerical integration stages; no
        Langevin correction or fresh per-step noise is used.
    """
    method = str(method).lower()
    if method != "heun":
        raise ValueError(
            f"Only the 'heun' ODE method is implemented, got {method!r}."
        )

    steps = sde.N if steps is None else int(steps)
    if steps <= 0:
        raise ValueError(f"ODE steps must be positive, got {steps}.")
    if not 0.0 < float(eps) < float(sde.T):
        raise ValueError(
            f"ODE eps must satisfy 0 < eps < {sde.T}, got {eps}."
        )

    rsde = sde.reverse(score_fn, probability_flow=True)

    def _match_layout(value, reference, name):
        if value is None:
            return None

        value = value.to(device=reference.device, dtype=reference.dtype)
        if reference.ndim == 4 and value.ndim == 3:
            value = value.unsqueeze(1)
        elif (
            reference.ndim == 3
            and value.ndim == 4
            and value.shape[1] == 1
        ):
            value = value.squeeze(1)

        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"ODE {name} shape must match the sampled state: "
                f"{name}={tuple(value.shape)}, state={tuple(reference.shape)}"
            )
        return value

    reference = y
    observed = _match_layout(y, reference, "conditioning")
    observed_mask = _match_layout(mask, reference, "mask")
    if observed_mask is not None:
        observed_mask = observed_mask.clamp(0.0, 1.0)
        unknown_mask = 1.0 - observed_mask
    else:
        unknown_mask = None

    def _expand_batch_coefficient(value, reference):
        """Broadcast a scalar or per-example value over a sampled state."""
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand(reference.shape[0])
        else:
            value = value.reshape(reference.shape[0])
        return value.reshape(
            reference.shape[0],
            *([1] * (reference.ndim - 1)),
        )

    def observed_at(t, observed_noise):
        """Return the known region with the training-time noise convention."""
        if on_noisy_masked_melspec:
            return observed

        mean, std = sde.marginal_prob(observed, None, t)
        std = _expand_batch_coefficient(std, observed)
        return mean + std * observed_noise

    def project_observed(x, t, observed_noise):
        """Apply the known-region boundary condition at continuous time t."""
        if observed_mask is None:
            return x
        observed_t = observed_at(t, observed_noise)
        return observed_mask * observed_t + unknown_mask * x

    def project_clean_observed(x):
        """Restore exact known values after reaching the clean endpoint."""
        if observed_mask is None:
            return x
        return observed_mask * observed + unknown_mask * x

    def drift_fn(x, t, observed_noise):
        """Evaluate the probability-flow velocity on the complete mel."""
        x_full = project_observed(x, t, observed_noise)
        drift = rsde.sde(x_full, observed, t)[0]
        if unknown_mask is not None:
            # The checkpoint can be trained with loss only on the missing
            # region. Its score in the known region must not drive the state.
            drift = unknown_mask * drift
        return drift

    def denoise_update_fn(x, t, observed_noise):
        """Use Tweedie's formula to estimate x_0 from the state at eps."""
        x_full = project_observed(x, t, observed_noise)
        score = score_fn(x_full, observed, t)
        std = sde.marginal_prob(x_full, None, t)[1]

        # VP has mean alpha(t) * x_0; VE has mean x_0 (alpha == 1).
        if hasattr(sde, "alpha"):
            alpha = sde.alpha(t)
        else:
            alpha = torch.ones_like(std)

        coefficient_shape = (
            x.shape[0],
            *([1] * (x.ndim - 1)),
        )
        alpha = alpha.to(x).reshape(coefficient_shape)
        std = std.to(x).reshape(coefficient_shape)
        x0 = (x_full + std.square() * score) / alpha
        return project_clean_observed(x0)

    def ode_sampler(z=None, observed_noise=None):
        """Integrate the probability-flow ODE from ``sde.T`` to ``eps``.

        Args:
            z: Optional initial latent. Supplying it makes comparisons and
                deterministic regression tests straightforward.
            observed_noise: Optional fixed noise for the observed-region
                forward marginal. It is sampled once when omitted and reused
                at every Heun stage; no fresh stochastic noise is injected
                during integration.

        Returns:
            The generated sample and the number of score evaluations.
        """
        with torch.no_grad():
            if z is None:
                x = sde.prior_sampling(y.shape, y)
            else:
                if tuple(z.shape) != tuple(y.shape):
                    raise ValueError(
                        "ODE latent shape must match the conditioning mel: "
                        f"z={tuple(z.shape)}, y={tuple(y.shape)}"
                    )
                x = z

            x = x.to(device=y.device, dtype=y.dtype)
            batch_size = x.shape[0]
            initial_t = torch.full(
                (batch_size,),
                float(sde.T),
                device=x.device,
                dtype=x.dtype,
            )

            if observed_mask is not None and not on_noisy_masked_melspec:
                if observed_noise is None:
                    observed_noise = torch.randn_like(observed)
                else:
                    observed_noise = _match_layout(
                        observed_noise,
                        observed,
                        "observed noise",
                    )
            else:
                observed_noise = None

            x = project_observed(x, initial_t, observed_noise)

            timesteps = torch.linspace(
                float(sde.T),
                float(eps),
                steps + 1,
                device=x.device,
                dtype=x.dtype,
            )

            for current_t, next_t in zip(timesteps[:-1], timesteps[1:]):
                dt = next_t - current_t  # Negative: generation runs T -> eps.
                vec_t = current_t.expand(batch_size)
                vec_next_t = next_t.expand(batch_size)

                # Heun predictor: an Euler proposal at the destination time.
                k1 = drift_fn(x, vec_t, observed_noise)
                x_euler = project_observed(
                    x + dt * k1,
                    vec_next_t,
                    observed_noise,
                )

                # Heun corrector: average the velocities at both endpoints.
                k2 = drift_fn(x_euler, vec_next_t, observed_noise)
                x = project_observed(
                    x + 0.5 * dt * (k1 + k2),
                    vec_next_t,
                    observed_noise,
                )

            nfe = 2 * steps
            if denoise:
                vec_eps = timesteps[-1].expand(batch_size)
                x = denoise_update_fn(x, vec_eps, observed_noise)
                nfe += 1

            return x, nfe

    return ode_sampler

def get_sb_sampler(sde, model, y, eps=1e-4, n_steps=50, sampler_type="ode", **kwargs):
    # adapted from https://github.com/NVIDIA/NeMo/blob/78357ae99ff2cf9f179f53fbcb02c88a5a67defb/nemo/collections/audio/parts/submodules/schroedinger_bridge.py#L382
    def sde_sampler():
        """The SB-SDE sampler function."""
        with torch.no_grad():
            xt = y[:, [0], :, :] # special case for storm_2ch
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / (alpha_prev * sigma_prev**2 + sde.eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + sde.eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_z = weight_z[:, None, None, None]

                # Random sample
                z_norm = torch.randn_like(xt)
                
                if t == time_steps[-1]:
                    weight_z = 0.0

                # Update state: weighted sum of previous state, current estimate and noise
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps

    def ode_sampler():
        """The SB-ODE sampler function."""
        with torch.no_grad():
            xt = y
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = model(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bart / (alpha_prev * sigma_prev * sigma_bar_prev + sde.eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_T**2 + sde.eps)
                    * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + sde.eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_T * sigma_T**2 + sde.eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + sde.eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_prior_mean = weight_prior_mean[:, None, None, None]

                # Update state: weighted sum of previous state, current estimate and prior
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_prior_mean * y

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, n_steps
    
    if sampler_type == "sde":
        return sde_sampler
    elif sampler_type == "ode":
        return ode_sampler
    else:
        raise ValueError("Invalid type. Choose 'ode' or 'sde'.")
