import abc
import torch

from sgmse import sdes
from sgmse.util.registry import Registry


CorrectorRegistry = Registry("Corrector")


def _batch_coefficient(value, reference):
    """Broadcast a scalar or per-batch coefficient over ``reference``."""
    if not torch.is_tensor(value):
        value = torch.as_tensor(
            value,
            device=reference.device,
            dtype=reference.dtype,
        )
    else:
        value = value.to(device=reference.device, dtype=reference.dtype)

    if value.ndim == 0:
        value = value.expand(reference.shape[0])
    elif value.numel() == 1 and reference.shape[0] != 1:
        value = value.reshape(1).expand(reference.shape[0])
    else:
        value = value.reshape(reference.shape[0])

    return value.reshape(
        reference.shape[0],
        *([1] * (reference.ndim - 1)),
    )


def _unknown_region(mask, reference):
    """Return a mask shaped like ``reference`` with 1 in generated regions."""
    if mask is None:
        return torch.ones_like(reference)

    observed = mask.to(device=reference.device, dtype=reference.dtype)

    # Match the sampler's supported [B, F, T] / [B, C, F, T] layouts.
    if reference.ndim == 4 and observed.ndim == 3:
        observed = observed.unsqueeze(1)
    elif (
        reference.ndim == 3
        and observed.ndim == 4
        and observed.shape[1] == 1
    ):
        observed = observed.squeeze(1)

    if tuple(observed.shape) != tuple(reference.shape):
        raise ValueError(
            "Corrector mask shape must match the sampled state: "
            f"mask={tuple(observed.shape)}, state={tuple(reference.shape)}"
        )

    # Project convention: mask == 1 is observed, mask == 0 is unknown.
    return (1.0 - observed).clamp(0.0, 1.0)


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps, mask=None):
        super().__init__()
        self.rsde = sde.reverse(score_fn)
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps
        self.mask = mask

    @abc.abstractmethod
    def update_fn(self, x, y, t, *args):
        """One update of the corrector.

        Args:
            x: A PyTorch tensor representing the current state
            t: A PyTorch tensor representing the current time step.
            *args: Possibly additional arguments, in particular `y` for OU processes

        Returns:
            x: A PyTorch tensor of the next state.
            x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@CorrectorRegistry.register(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps, mask=None):
        super().__init__(sde, score_fn, snr, n_steps, mask=mask)
        self.score_fn = score_fn
        self.n_steps = n_steps
        self.snr = snr

    def update_fn(self, x, y, t, *args):
        target_snr = self.snr
        x_mean = x
        unknown = _unknown_region(self.mask, x)
        has_unknown = unknown.reshape(x.shape[0], -1).sum(dim=-1) > 0

        for _ in range(self.n_steps):
            grad = self.score_fn(x, y, t, *args)
            noise = torch.randn_like(x)

            # The score sees the complete state, but the Langevin move and its
            # target-SNR calculation are restricted to the generated region.
            grad = grad * unknown
            noise = noise * unknown
            grad_norm = torch.norm(
                grad.reshape(grad.shape[0], -1),
                dim=-1,
            ).clamp_min(1e-12)
            noise_norm = torch.norm(
                noise.reshape(noise.shape[0], -1),
                dim=-1,
            )
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2
            step_size = torch.where(
                has_unknown,
                step_size,
                torch.zeros_like(step_size),
            )
            step = _batch_coefficient(step_size, x)
            x_mean = x + step * grad
            x = x_mean + noise * torch.sqrt(step * 2)

        return x, x_mean


@CorrectorRegistry.register(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2."""
    def __init__(self, sde, score_fn, snr, n_steps, mask=None):
        super().__init__(sde, score_fn, snr, n_steps, mask=mask)
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    def update_fn(self, x, y, t, *args):
        n_steps = self.n_steps
        target_snr = self.snr
        std = self.sde.marginal_prob(x, y, t, *args)[1]
        x_mean = x
        unknown = _unknown_region(self.mask, x)
        has_unknown = unknown.reshape(x.shape[0], -1).sum(dim=-1) > 0

        for _ in range(n_steps):
            grad = self.score_fn(x, y, t, *args)
            noise = torch.randn_like(x)

            # Anneal as before, but only update and inject noise in the gap.
            grad = grad * unknown
            noise = noise * unknown
            step_size = (target_snr * std) ** 2 * 2
            step = _batch_coefficient(step_size, x)
            step = step * _batch_coefficient(has_unknown, x)
            x_mean = x + step * grad
            x = x_mean + noise * torch.sqrt(step * 2)

        return x, x_mean


@CorrectorRegistry.register(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, *args, **kwargs):
        self.snr = 0
        self.n_steps = 0
        pass

    def update_fn(self, x, y, t, *args):
        # Accept `y` for API consistency with other correctors (even if unused).
        return x, x
