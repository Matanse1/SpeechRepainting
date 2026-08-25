import unittest

import torch

from sampling import get_ode_sampler


class _ConstantReverseODE:
    def __init__(self, velocity):
        self.velocity = velocity

    def sde(self, x, y, t):
        del y, t
        drift = torch.full_like(x, self.velocity)
        diffusion = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return drift, diffusion


class _FakeSDE:
    N = 10
    T = 1.0

    def __init__(self, velocity=1.0, std=0.0, alpha=1.0):
        self.velocity = velocity
        self.std = std
        self.alpha_value = alpha
        self.used_probability_flow = None

    def reverse(self, score_fn, probability_flow=False):
        del score_fn
        self.used_probability_flow = probability_flow
        return _ConstantReverseODE(self.velocity)

    def prior_sampling(self, shape, y=None):
        del y
        return torch.zeros(shape)

    def marginal_prob(self, x, y, t):
        del y
        std = torch.full_like(t, self.std)
        return x, std

    def alpha(self, t):
        return torch.full_like(t, self.alpha_value)


class _TimeDependentMarginalSDE(_FakeSDE):
    def marginal_prob(self, x, y, t):
        del y
        coefficient = t.reshape(t.shape[0], *([1] * (x.ndim - 1)))
        return coefficient * x, 1.0 - t


class HeunODESamplerTest(unittest.TestCase):
    def test_heun_integrates_backward_and_preserves_observed_region(self):
        sde = _FakeSDE(velocity=1.0)
        y = torch.tensor([[[4.0, 5.0, 6.0]]])
        mask = torch.tensor([[[1.0, 0.0, 1.0]]])
        z = torch.zeros_like(y)

        sampler = get_ode_sampler(
            sde=sde,
            score_fn=lambda x, y, t: torch.zeros_like(x),
            y=y,
            mask=mask,
            method="heun",
            steps=3,
            eps=0.1,
            denoise=False,
        )
        sample, nfe = sampler(z=z)

        expected = torch.tensor([[[4.0, -0.9, 6.0]]])
        torch.testing.assert_close(sample, expected)
        self.assertEqual(nfe, 6)
        self.assertTrue(sde.used_probability_flow)

    def test_fully_noised_training_uses_fixed_observed_noise_path(self):
        sde = _TimeDependentMarginalSDE(velocity=1.0)
        y = torch.tensor([[[4.0, 5.0, 6.0]]])
        mask = torch.tensor([[[1.0, 0.0, 1.0]]])
        z = torch.zeros_like(y)
        observed_noise = torch.tensor([[[2.0, 3.0, 4.0]]])

        sampler = get_ode_sampler(
            sde=sde,
            score_fn=lambda x, y, t: torch.zeros_like(x),
            y=y,
            mask=mask,
            on_noisy_masked_melspec=False,
            method="heun",
            steps=3,
            eps=0.1,
            denoise=False,
        )
        sample, nfe = sampler(z=z, observed_noise=observed_noise)

        # At eps=0.1, y_t = 0.1*y + 0.9*observed_noise. The
        # missing bin follows the backward constant-velocity ODE.
        expected = torch.tensor([[[2.2, -0.9, 4.2]]])
        torch.testing.assert_close(sample, expected)
        self.assertEqual(nfe, 6)

    def test_denoising_restores_clean_observed_region(self):
        sde = _TimeDependentMarginalSDE(velocity=0.0, alpha=1.0)
        y = torch.tensor([[[4.0, 5.0, 6.0]]])
        mask = torch.tensor([[[1.0, 0.0, 1.0]]])

        sampler = get_ode_sampler(
            sde=sde,
            score_fn=lambda x, y, t: torch.zeros_like(x),
            y=y,
            mask=mask,
            on_noisy_masked_melspec=False,
            steps=2,
            eps=0.1,
            denoise=True,
        )
        sample, _ = sampler(
            z=torch.zeros_like(y),
            observed_noise=torch.ones_like(y),
        )

        torch.testing.assert_close(sample[..., 0], y[..., 0])
        torch.testing.assert_close(sample[..., 2], y[..., 2])

    def test_fixed_latent_is_deterministic(self):
        sde = _FakeSDE(velocity=0.25)
        y = torch.zeros(2, 3, 4)
        z = torch.randn_like(y)
        sampler = get_ode_sampler(
            sde=sde,
            score_fn=lambda x, y, t: torch.zeros_like(x),
            y=y,
            method="heun",
            steps=4,
            eps=0.2,
            denoise=False,
        )

        first, _ = sampler(z=z)
        second, _ = sampler(z=z)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_denoising_counts_and_uses_final_score_evaluation(self):
        sde = _FakeSDE(velocity=0.0, std=0.5, alpha=0.8)
        y = torch.zeros(1, 2, 2)
        z = torch.ones_like(y)
        score_calls = 0

        def score_fn(x, y, t):
            nonlocal score_calls
            del y, t
            score_calls += 1
            return torch.ones_like(x)

        sampler = get_ode_sampler(
            sde=sde,
            score_fn=score_fn,
            y=y,
            method="heun",
            steps=2,
            eps=0.1,
            denoise=True,
        )
        sample, nfe = sampler(z=z)

        torch.testing.assert_close(sample, torch.full_like(sample, 1.5625))
        self.assertEqual(score_calls, 1)
        self.assertEqual(nfe, 5)

    def test_rejects_any_method_other_than_heun(self):
        with self.assertRaisesRegex(ValueError, "Only the 'heun'"):
            get_ode_sampler(
                sde=_FakeSDE(),
                score_fn=lambda x, y, t: x,
                y=torch.zeros(1, 2, 2),
                method="euler",
            )


if __name__ == "__main__":
    unittest.main()
