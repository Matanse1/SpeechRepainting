"""
Abstract SDE classes, reverse-time SDE construction,
and two concrete SDE implementations used for speech enhancement.

This version is heavily commented for understanding.

Originally adapted from:
https://github.com/yang-song/score_sde_pytorch
"""

import abc                     # For abstract base classes / abstract methods
import warnings                # For warning messages when shapes do not match

import numpy as np             # Used mainly for scalar math / logs / sqrt
import torch                   # Main tensor library

from sgmse.util.tensors import batch_broadcast
from sgmse.util.registry import Registry


# -----------------------------------------------------------------------------
# Registry:
# This allows looking up SDE classes by name, e.g. "ouve" or "sbve".
# -----------------------------------------------------------------------------
SDERegistry = Registry("SDE")

# =============================================================================
# Abstract base class: SDE
# =============================================================================
class SDE(abc.ABC):
    """
    Abstract SDE class.

    Any specific SDE implementation (OUVE, VE, VP, SBVE, etc.)
    should inherit from this class and implement the required methods.

    All functions are designed to work on mini-batches:
    - x is a batch of noisy signals
    - y is the conditioning signal (e.g. observed noisy speech)
    - t is a batch of time values, one scalar per sample in the batch
    """

    def __init__(self, N):
        """
        Construct an SDE.

        Args:
            N: number of discretization time steps
               used in numerical sampling.
        """
        super().__init__()
        self.N = N

    # -------------------------------------------------------------------------
    # T = end time of the SDE
    # Usually diffusion runs from t=0 to t=T (here typically T=1).
    # -------------------------------------------------------------------------
    @property
    @abc.abstractmethod
    def T(self):
        """End time (final time horizon) of the SDE."""
        pass

    # -------------------------------------------------------------------------
    # sde(...) defines the forward SDE:
    # dx = drift(x,t) dt + diffusion(t) dW
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def sde(self, x, y, t, *args):
        """
        Return the drift and diffusion terms of the forward SDE.

        Args:
            x: current state x_t
            y: conditioning / target anchor (depends on the SDE)
            t: current time
        Returns:
            drift, diffusion
        """
        pass

    # -------------------------------------------------------------------------
    # marginal_prob(...) gives the distribution of x_t given x_0
    # returns mean and std for q(x_t | x_0, y)
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def marginal_prob(self, x, y, t, *args):
        """
        Return parameters of the marginal distribution p_t(x | args).

        In most diffusion implementations, this returns:
            mean_t, std_t

        so that:
            x_t = mean_t + std_t * noise
        """
        pass

    # -------------------------------------------------------------------------
    # Sample from the prior / terminal distribution p_T
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """
        Sample from the terminal / prior distribution p_T(x).

        Args:
            shape: desired tensor shape
        Returns:
            x_T sample
        """
        pass

    # -------------------------------------------------------------------------
    # Log-density of p_T(z)
    # Needed for probability flow ODE likelihood computations.
    # Often not implemented in many practical projects.
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def prior_logp(self, z):
        """
        Compute log-density under the prior distribution.

        Useful for exact likelihood / probability flow ODE methods.

        Args:
            z: latent code / terminal sample
        Returns:
            log probability density
        """
        pass

    # -------------------------------------------------------------------------
    # Add argparse CLI arguments required by this specific SDE
    # -------------------------------------------------------------------------
    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add arguments needed to construct this SDE
        into an argparse parser.
        """
        pass

    # -------------------------------------------------------------------------
    # Default discretization:
    # Euler-Maruyama step:
    #   x_{i+1} = x_i + f_i + G_i * z_i
    #
    # where:
    #   f_i = drift * dt
    #   G_i = diffusion * sqrt(dt)
    # -------------------------------------------------------------------------
    def discretize(self, x, y, t, stepsize):
        """
        Discretize the SDE for numerical sampling.

        Default: Euler-Maruyama.

        Args:
            x: current state
            y: conditioning signal
            t: current time
            stepsize: dt

        Returns:
            f, G
            where:
              f = deterministic increment
              G = noise scale for this step
        """
        dt = stepsize

        # Get instantaneous drift and diffusion
        drift, diffusion = self.sde(x, y, t)

        # Deterministic part scales with dt
        f = drift * dt

        # Stochastic noise part scales with sqrt(dt)
        G = diffusion * torch.sqrt(dt)

        return f, G

    # -------------------------------------------------------------------------
    # reverse(...) constructs the reverse-time SDE or ODE
    #
    # This is one of the key ideas in score-based diffusion models:
    # if you know the score ∇_x log p_t(x|y), you can reverse the forward noise
    # process and generate / denoise samples.
    #
    # probability_flow=False -> reverse SDE (still stochastic)
    # probability_flow=True  -> probability flow ODE (deterministic)
    # -------------------------------------------------------------------------
    def reverse(oself, score_model, probability_flow=False):
        """
        Create the reverse-time SDE/ODE from the forward SDE.

        Args:
            score_model:
                function that takes (x, y, t, *args)
                and returns the score:
                    score = ∇_x log p_t(x | y)
            probability_flow:
                - False: reverse SDE
                - True:  probability flow ODE (deterministic)
        """
        # Save values from the original SDE instance
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # ---------------------------------------------------------------------
        # Dynamically define a reverse-time SDE class that inherits
        # from the same class as the original SDE.
        # ---------------------------------------------------------------------
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, y, t, *args):
                """
                Return the reverse-time drift and diffusion.
                """
                rsde_parts = self.rsde_parts(x, y, t, *args)

                total_drift = rsde_parts["total_drift"]
                diffusion = rsde_parts["diffusion"]

                return total_drift, diffusion

            def rsde_parts(self, x, y, t, *args):
                """
                Compute all components of the reverse SDE.

                Forward SDE:
                    dx = f(x,t) dt + g(t) dW

                Reverse SDE (roughly):
                    dx = [f(x,t) - g(t)^2 * score(x,t)] dt + g(t) dW_bar

                Probability flow ODE:
                    dx = [f(x,t) - 0.5 * g(t)^2 * score(x,t)] dt
                """
                # Forward drift and diffusion
                sde_drift, sde_diffusion = sde_fn(x, y, t, *args)

                # Score estimated by neural network
                score = score_model(x, y, t, *args)

                # score_drift is the correction term that "reverses" the process
                # sde_diffusion is shape [B], while score is [B, C, F, T],
                # so we expand dimensions to broadcast properly.
                score_drift = -sde_diffusion[:, None, None] ** 2 * score * (
                    0.5 if self.probability_flow else 1.0
                )

                # In probability flow ODE, diffusion is set to zero (deterministic)
                diffusion = (
                    torch.zeros_like(sde_diffusion)
                    if self.probability_flow
                    else sde_diffusion
                )

                # Total reverse drift = forward drift + score correction
                total_drift = sde_drift + score_drift

                return {
                    "total_drift": total_drift,
                    "diffusion": diffusion,
                    "sde_drift": sde_drift,
                    "sde_diffusion": sde_diffusion,
                    "score_drift": score_drift,
                    "score": score,
                }

            def discretize(self, x, y, t, stepsize):
                """
                Discretized reverse-time update rule.

                If forward discretization is:
                    x_{i+1} = x_i + f + G z

                then reverse uses:
                    rev_f = f - G^2 * score(...)
                and possibly rev_G = 0 for probability flow ODE
                """
                f, G = discretize_fn(x, y, t, stepsize)

                rev_f = f - G[:, None, None] ** 2 * score_model(x, y, t) * (
                    0.5 if self.probability_flow else 1.0
                )

                rev_G = torch.zeros_like(G) if self.probability_flow else G

                return rev_f, rev_G

        # Return an instance of the reverse SDE/ODE
        return RSDE()

    # -------------------------------------------------------------------------
    # copy() must be implemented by each concrete SDE class
    # -------------------------------------------------------------------------
    @abc.abstractmethod
    def copy(self):
        pass


# # =============================================================================
# # OUVESDE = Ornstein-Uhlenbeck Variance Exploding SDE
# # =============================================================================
# @SDERegistry.register("ouve")
# class OUVESDE(SDE):
#     """
#     Ornstein-Uhlenbeck Variance Exploding SDE.

#     Forward process:
#         dx = theta * (y - x) dt + sigma(t) dW
    
#     Variance Exploding" means sigma(t) grows over time:
#         sigma(t) = sigma_min (sigma_max / sigma_min)^t    

#     Interpretation:
#     - The drift pulls x toward y (mean-reverting behavior).
#     - The diffusion injects noise with a time-dependent scale sigma(t).
#     - Since sigma(t) grows with t, noise "explodes" over time.
#     """

#     @staticmethod
#     def add_argparse_args(parser):
#         """
#         CLI arguments for constructing this SDE from command line.
#         """
#         parser.add_argument(
#             "--theta",
#             type=float,
#             default=1.5,
#             help="Stiffness of the OU process. Higher = stronger pull toward y."
#         )
#         parser.add_argument(
#             "--sigma-min",
#             type=float,
#             default=0.05,
#             help="Minimum diffusion scale."
#         )
#         parser.add_argument(
#             "--sigma-max",
#             type=float,
#             default=0.5,
#             help="Maximum diffusion scale."
#         )
#         parser.add_argument(
#             "--N",
#             type=int,
#             default=30,
#             help="Number of discretization steps."
#         )
#         parser.add_argument(
#             "--sampler_type",
#             type=str,
#             default="pc",
#             help="Sampler type, e.g. predictor-corrector."
#         )
#         return parser

#     def __init__(self, theta, sigma_min, sigma_max, N=30, sampler_type="pc", **ignored_kwargs):
#         """
#         Construct the OUVE SDE.

#         Args:
#             theta: how strongly x is pulled toward y
#             sigma_min: smallest diffusion scale
#             sigma_max: largest diffusion scale
#             N: number of time steps
#             sampler_type: metadata about which sampler is preferred
#         """
#         super().__init__(N)

#         self.theta = theta
#         self.sigma_min = sigma_min
#         self.sigma_max = sigma_max

#         # logsig = log(sigma_max / sigma_min)
#         # useful because sigma(t) is exponential in t
#         self.logsig = np.log(self.sigma_max / self.sigma_min)

#         self.N = N
#         self.sampler_type = sampler_type

#     def copy(self):
#         """
#         Return a duplicate with the same hyperparameters.
#         """
#         return OUVESDE(
#             self.theta,
#             self.sigma_min,
#             self.sigma_max,
#             N=self.N,
#             sampler_type=self.sampler_type
#         )

#     @property
#     def T(self):
#         """
#         End time horizon of the diffusion process.
#         """
#         return 1

#     def sde(self, x, y, t):
#         """
#         Return the forward drift and diffusion at time t.

#         Drift:
#             theta * (y - x)

#         This pulls x toward y.
#         If x is far from y, drift is larger.

#         Diffusion:
#             sigma(t) * sqrt(2 * logsig)

#         where sigma(t) grows exponentially from sigma_min to sigma_max.
#         """
#         # Mean-reverting drift toward y
#         drift = self.theta * (y - x)

#         # Base exponential schedule for sigma(t)
#         sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t

#         # Extra sqrt(2*logsig) factor:
#         # This is included so that the final perturbation kernel has the desired variance.
#         diffusion = sigma * np.sqrt(2 * self.logsig)

#         return drift, diffusion

#     def _mean(self, x0, y, t):
#         """
#         Closed-form mean of x_t given x0 and y.

#         Since drift pulls toward y, the mean becomes:
#             exp(-theta t) * x0 + (1 - exp(-theta t)) * y

#         So:
#         - at t=0, mean = x0
#         - as t grows, mean moves toward y
#         """
#         theta = self.theta

#         # Shape becomes [B,1,1,1] to broadcast over tensor dimensions
#         exp_interp = torch.exp(-theta * t)[:, None, None]

#         return exp_interp * x0 + (1 - exp_interp) * y

#     def alpha(self, t):
#         """
#         Useful scalar factor:
#             alpha(t) = exp(-theta t)

#         This tells you how much of x0 remains in the mean.
#         """
#         return torch.exp(-self.theta * t)

#     def _std(self, t):
#         """
#         Closed-form standard deviation of x_t.

#         This comes from solving the variance ODE exactly.
#         The formula is derived from integrating the effect of
#         the time-varying diffusion under the OU dynamics.
#         """
#         sigma_min = self.sigma_min
#         theta = self.theta
#         logsig = self.logsig

#         return torch.sqrt(
#             (
#                 sigma_min ** 2
#                 * torch.exp(-2 * theta * t)
#                 * (torch.exp(2 * (theta + logsig) * t) - 1)
#                 * logsig
#             )
#             /
#             (theta + logsig)
#         )

#     def marginal_prob(self, x0, y, t):
#         """
#         Return the marginal distribution parameters:
#             mean_t, std_t

#         so x_t can be written as:
#             x_t = mean_t + std_t * noise
#         """
#         return self._mean(x0, y, t), self._std(t)

#     def prior_sampling(self, shape, y):
#         """
#         Sample x_T from the terminal distribution.

#         For this SDE, at t=T the distribution is centered around y:
#             x_T ~ N(y, std(T)^2 I)

#         So we sample:
#             x_T = y + noise * std(T)
#         """
#         if shape != y.shape:
#             warnings.warn(
#                 f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape."
#             )

#         # std evaluated at t=1 for each batch item
#         std = self._std(torch.ones((y.shape[0],), device=y.device))

#         # Broadcast std over the remaining tensor dimensions
#         x_T = y + torch.randn_like(y) * std[:, None, None]

#         return x_T

#     def prior_logp(self, z):
#         """
#         Not implemented.
#         """
#         raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


# # =============================================================================
# # SBVESDE = Schrödinger Bridge Variance Exploding SDE
# # =============================================================================
# @SDERegistry.register("sbve")
# class SBVESDE(SDE):
#     """
#     Schrödinger Bridge Variance Exploding SDE.

#     This follows the formulation in:
#     Jukić et al., "Schrödinger Bridge for Generative Speech Enhancement", 2024.

#     Unlike OUVE:
#     - drift is zero
#     - diffusion grows as k^t

#     The bridge formulation defines a forward process that connects x0 and y
#     in a structured probabilistic way.
#     """

#     @staticmethod
#     def add_argparse_args(parser):
#         """
#         CLI arguments for this SDE.
#         """
#         parser.add_argument(
#             "--N",
#             type=int,
#             default=50,
#             help="Number of discretization steps."
#         )
#         parser.add_argument(
#             "--k",
#             type=float,
#             default=2.6,
#             help="Controls exponential growth of diffusion."
#         )
#         parser.add_argument(
#             "--c",
#             type=float,
#             default=0.4,
#             help="Diffusion scale parameter."
#         )
#         parser.add_argument(
#             "--eps",
#             type=float,
#             default=1e-8,
#             help="Small constant for numerical stability."
#         )
#         parser.add_argument(
#             "--sampler_type",
#             type=str,
#             default="ode",
#             help="Preferred sampler type (often ODE for SBVE)."
#         )
#         return parser

#     def __init__(self, k, c, N=50, eps=1e-8, sampler_type="ode", **ignored_kwargs):
#         """
#         Construct the SBVE SDE.

#         Args:
#             k: controls how fast diffusion grows with time
#             c: diffusion scaling factor
#             N: discretization steps
#             eps: numerical stability constant
#             sampler_type: metadata about preferred sampler
#         """
#         super().__init__(N)

#         self.k = k
#         self.c = c
#         self.N = N
#         self.eps = eps
#         self.sampler_type = sampler_type

#     def copy(self):
#         """
#         Return a duplicate of this SDE.
#         """
#         return SBVESDE(self.k, self.c, N=self.N)

#     @property
#     def T(self):
#         """
#         End time horizon.
#         """
#         return 1

#     def sde(self, x, y, t):
#         """
#         Return forward drift and diffusion.

#         For SBVE (as written here):
#             drift f = 0
#             diffusion g(t) = sqrt(c) * k^t

#         So the process has no deterministic drift,
#         only time-varying stochastic diffusion.
#         """
#         f = 0.0

#         # NOTE:
#         # torch.sqrt(torch.tensor(self.c)) creates a CPU tensor by default.
#         # In practice, it is often cleaner to keep this on the same device as t.
#         # But this is the original structure.
#         g = torch.sqrt(torch.tensor(self.c)) * self.k ** t

#         return f, g

#     def _sigmas_alphas(self, t):
#         """
#         Compute several helper quantities used in the Schrödinger Bridge formulas.

#         Returns:
#             sigma_t     : std at time t
#             sigma_T     : std at final time T
#             sigma_bart  : conditional std-like term
#             alpha_t     : alpha at time t (here always 1)
#             alpha_T     : alpha at time T (here always 1)
#             alpha_bart  : alpha_t / alpha_T
#         """
#         # In this specific formulation, alpha is constant = 1
#         alpha_t = torch.ones_like(t)
#         alpha_T = torch.ones_like(t)

#         # Closed-form sigma(t) from the paper (Table 1)
#         sigma_t = torch.sqrt(
#             (self.c * (self.k ** (2 * t) - 1.0))
#             / (2 * torch.log(torch.tensor(self.k)))
#         )

#         # Same, but at final time T
#         sigma_T = torch.sqrt(
#             (self.c * (self.k ** (2 * self.T) - 1.0))
#             / (2 * torch.log(torch.tensor(self.k)))
#         )

#         # "bar" quantities used below Eq. (9) in the paper
#         alpha_bart = alpha_t / (alpha_T + self.eps)

#         # Remaining noise scale from time t to T
#         sigma_bart = torch.sqrt(sigma_T ** 2 - sigma_t ** 2 + self.eps)

#         return sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart

#     def _mean(self, x0, y, t):
#         """
#         Closed-form mean of x_t under the SB bridge.

#         The mean is a weighted combination of:
#         - x0
#         - y

#         where the weights depend on t through sigma_t / sigma_T / sigma_bart.
#         """
#         sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

#         # Weight for x0
#         w_xt = alpha_t * sigma_bart ** 2 / (sigma_T ** 2 + self.eps)

#         # Weight for y
#         w_yt = alpha_bart * sigma_t ** 2 / (sigma_T ** 2 + self.eps)

#         # Weighted interpolation between x0 and y
#         mu = (
#             w_xt[:, None, None] * x0
#             + w_yt[:, None, None] * y
#         )

#         return mu

#     def _std(self, t):
#         """
#         Closed-form standard deviation of x_t for the bridge.
#         """
#         sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

#         sigma_xt = (alpha_t * sigma_bart * sigma_t) / (sigma_T + self.eps)

#         return sigma_xt

#     def marginal_prob(self, x0, y, t):
#         """
#         Return mean and std of x_t.
#         """
#         return self._mean(x0, y, t), self._std(t)

#     def prior_sampling(self, shape, y):
#         """
#         Sample x_T from terminal distribution.

#         In this formulation, x_T is set directly to y.

#         That means the terminal distribution is effectively deterministic:
#             x_T = y

#         This is very different from the usual Gaussian terminal prior
#         in standard diffusion models.
#         """
#         if shape != y.shape:
#             warnings.warn(
#                 f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape."
#             )

#         x_T = y
#         return x_T

#     def prior_logp(self, z):
#         """
#         Not implemented.
#         """
#         raise NotImplementedError("prior_logp for SBVE SDE not yet implemented!")


# =============================================================================
# VPSDE = Variance Preserving SDE (continuous analog of DDPM)
# =============================================================================
@SDERegistry.register("vp")
class VPSDE(SDE):
    """
    Variance Preserving SDE.

    This is the continuous-time analog of DDPM (Ho et al., 2020),
    as formulated in Song et al., "Score-Based Generative Modeling
    through Stochastic Differential Equations", ICLR 2021.

    Forward process:
        dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dW

    Interpretation:
    - The drift shrinks x toward zero (not toward y).
    - The diffusion injects noise scaled by beta(t).
    - "Variance Preserving" means the total variance stays bounded:
      as x_0 shrinks, noise grows to compensate.
    - At t=T, x_T ~ N(0, I) (pure Gaussian noise).

    Unlike OUVE/SBVE, this SDE is unconditional (no y in the drift).
    y is only used as an optional conditioning input to the score model.
    """

    @staticmethod
    def add_argparse_args(parser):
        """
        CLI arguments for constructing this SDE from command line.
        """
        parser.add_argument(
            "--beta-min",
            type=float,
            default=0.1,
            help="Minimum noise schedule value (at t=0)."
        )
        parser.add_argument(
            "--beta-max",
            type=float,
            default=20.0,
            help="Maximum noise schedule value (at t=T)."
        )
        parser.add_argument(
            "--N",
            type=int,
            default=1000,
            help="Number of discretization steps."
        )
        parser.add_argument(
            "--sampler_type",
            type=str,
            default="pc",
            help="Sampler type, e.g. predictor-corrector."
        )
        return parser

    def __init__(self, beta_min, beta_max, N=1000, sampler_type="pc", **ignored_kwargs):
        """
        Construct the VP SDE.

        Args:
            beta_min: noise schedule value at t=0
            beta_max: noise schedule value at t=T
            N: number of discretization steps
            sampler_type: metadata about which sampler is preferred
        """
        super().__init__(N)

        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        """
        Return a duplicate with the same hyperparameters.
        """
        return VPSDE(
            self.beta_min,
            self.beta_max,
            N=self.N,
            sampler_type=self.sampler_type
        )

    @property
    def T(self):
        """
        End time horizon of the diffusion process.
        """
        return 1

    def _beta(self, t):
        """
        Linear noise schedule:
            beta(t) = beta_min + t * (beta_max - beta_min)

        This is the continuous analog of the discrete beta schedule in DDPM.
        At t=0: beta = beta_min (small noise)
        At t=T: beta = beta_max (large noise)
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _integral_beta(self, t):
        """
        Closed-form integral of beta(t) from 0 to t:
            int_0^t beta(s) ds = beta_min * t + 0.5 * (beta_max - beta_min) * t^2

        This is used to compute alpha(t) analytically without summing steps.
        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def alpha(self, t):
        """
        alpha(t) = exp(-0.5 * integral_beta(t))

        This tells you how much of x0 survives at time t.
        At t=0: alpha = 1 (x0 fully preserved)
        At t=T: alpha ~ 0 (x0 almost gone)
        """
        return torch.exp(-0.5 * self._integral_beta(t))

    def sde(self, x, y, t):
        """
        Return the forward drift and diffusion at time t.

        Drift:
            -0.5 * beta(t) * x

        This shrinks x toward zero over time.

        Diffusion:
            sqrt(beta(t))

        Note: y is accepted but not used in the drift.
        It may still be passed to the score model externally.
        """
        beta_t = self._beta(t)

        # Drift shrinks x toward zero
        # beta_t is shape [B], so we broadcast over [B, C, F, T]
        drift = -0.5 * beta_t[:, None, None] * x

        # Diffusion coefficient
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def _mean(self, x0, y, t):
        """
        Closed-form mean of x_t given x0:
            mean_t = alpha(t) * x0

        The mean decays toward zero as t grows.
        y is accepted for API consistency but is not used.
        """
        alpha_t = self.alpha(t)[:, None, None]
        return alpha_t * x0

    def _std(self, t):
        """
        Closed-form standard deviation of x_t:
            std_t = sqrt(1 - alpha(t)^2)

        "Variance Preserving" means:
            alpha(t)^2 + std(t)^2 = 1

        So as alpha decays, std grows to compensate.
        At t=0: std = 0
        At t=T: std ~ 1
        """
        alpha_t = self.alpha(t)
        return torch.sqrt(1 - alpha_t ** 2)

    def marginal_prob(self, x0, y, t):
        """
        Return the marginal distribution parameters:
            mean_t, std_t

        so x_t can be written as:
            x_t = mean_t + std_t * noise
            x_t = alpha(t) * x0 + sqrt(1 - alpha(t)^2) * noise
        """
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y=None):
        """
        Sample x_T from the terminal distribution.

        For VP SDE, the terminal distribution is standard Gaussian:
            x_T ~ N(0, I)

        This is very different from OUVE (centered at y)
        and SBVE (exactly y).
        y is accepted for API consistency but is not used.
        """
        return torch.randn(shape)

    def prior_logp(self, z):
        """
        Log-density of the standard Gaussian terminal distribution:
            log p_T(z) = -0.5 * ||z||^2 - 0.5 * D * log(2*pi)

        where D is the total number of dimensions.
        """
        shape = z.shape
        N = np.prod(shape[1:])  # number of dimensions (excluding batch)
        return -0.5 * N * np.log(2 * np.pi) - 0.5 * torch.sum(z ** 2, dim=(1, 2, 3))


# =============================================================================
# VESDE = Variance Exploding SDE (unconditional, no y)
# =============================================================================
@SDERegistry.register("ve")
class VESDE(SDE):
    """
    Variance Exploding SDE.

    As formulated in Song et al., "Score-Based Generative Modeling
    through Stochastic Differential Equations", ICLR 2021.

    Forward process:
        dx = sigma(t) * sqrt(d(sigma^2)/dt) dW
        (equivalently: dx = 0 dt + sigma(t) dW in simplified form)

    Interpretation:
    - The drift is zero: x0 is never attenuated (alpha(t) = 1 always).
    - The diffusion grows exponentially: sigma(t) = sigma_min * (sigma_max/sigma_min)^t
    - "Variance Exploding" means noise variance grows without bound.
    - At t=T, x_T ~ N(x0, sigma_max^2 * I), but sigma_max is so large
      that x0 is buried under noise.

    Unlike SBVE (which is also VE-type but uses a bridge to y),
    this version is unconditional: it does not enforce x_T = y.
    """

    @staticmethod
    def add_argparse_args(parser):
        """
        CLI arguments for constructing this SDE from command line.
        """
        parser.add_argument(
            "--sigma-min",
            type=float,
            default=0.01,
            help="Minimum noise level (at t=0)."
        )
        parser.add_argument(
            "--sigma-max",
            type=float,
            default=50.0,
            help="Maximum noise level (at t=T). Should be large enough to bury the signal."
        )
        parser.add_argument(
            "--N",
            type=int,
            default=1000,
            help="Number of discretization steps."
        )
        parser.add_argument(
            "--sampler_type",
            type=str,
            default="pc",
            help="Sampler type, e.g. predictor-corrector."
        )
        return parser

    def __init__(self, sigma_min, sigma_max, N=1000, sampler_type="pc", **ignored_kwargs):
        """
        Construct the VE SDE.

        Args:
            sigma_min: noise level at t=0 (small)
            sigma_max: noise level at t=T (large, should bury the signal)
            N: number of discretization steps
            sampler_type: metadata about which sampler is preferred
        """
        super().__init__(N)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        """
        Return a duplicate with the same hyperparameters.
        """
        return VESDE(
            self.sigma_min,
            self.sigma_max,
            N=self.N,
            sampler_type=self.sampler_type
        )

    @property
    def T(self):
        """
        End time horizon of the diffusion process.
        """
        return 1

    def _sigma(self, t):
        """
        Exponential noise schedule:
            sigma(t) = sigma_min * (sigma_max / sigma_min)^t

        At t=0: sigma = sigma_min (almost no noise)
        At t=T: sigma = sigma_max (signal buried)

        This is the same exponential structure as SBVE (with k = sigma_max/sigma_min),
        but without the bridge constraint that pins x_T to y.
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(self, x, y, t):
        """
        Return the forward drift and diffusion at time t.

        Drift:
            0  (no deterministic component)

        Diffusion:
            sigma(t) * sqrt(2 * log(sigma_max / sigma_min))

        The sqrt(2 * log(...)) factor ensures the variance of x_t
        matches sigma(t)^2 exactly (derived from Ito's formula).

        Note: y is accepted but not used.
        """
        sigma_t = self._sigma(t)

        # No drift: x0 is never attenuated
        drift = torch.zeros_like(x)

        # Diffusion grows exponentially
        logsig = np.log(self.sigma_max / self.sigma_min)
        diffusion = sigma_t * np.sqrt(2 * logsig)

        return drift, diffusion

    def _mean(self, x0, y, t):
        """
        Closed-form mean of x_t given x0:
            mean_t = x0

        Since there is no drift, the mean never moves away from x0.
        alpha(t) = 1 always.
        y is accepted for API consistency but is not used.
        """
        return x0

    def _std(self, t):
        """
        Closed-form standard deviation of x_t:
            std_t = sigma(t)

        The noise accumulates exponentially.
        At t=0: std ~ sigma_min (very small)
        At t=T: std = sigma_max (very large)
        """
        return self._sigma(t)

    def marginal_prob(self, x0, y, t):
        """
        Return the marginal distribution parameters:
            mean_t, std_t

        so x_t can be written as:
            x_t = x0 + sigma(t) * noise

        Note: alpha(t) = 1, so x0 is always fully present in the mean.
        The signal is not destroyed, just buried under growing noise.
        """
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y=None):
        """
        Sample x_T from the terminal distribution.

        For VE SDE, the terminal distribution is:
            x_T ~ N(0, sigma_max^2 * I)

        We approximate this as pure Gaussian scaled by sigma_max,
        since sigma_max is large enough that x0 is negligible.

        This is different from:
        - OUVE: x_T ~ N(y, std_T^2 I)  — centered at noisy speech
        - SBVE: x_T = y               — exactly noisy speech
        - VPSDE: x_T ~ N(0, I)        — standard Gaussian
        y is accepted for API consistency but is not used.
        """
        return torch.randn(shape) * self.sigma_max

    def prior_logp(self, z):
        """
        Log-density of the terminal Gaussian N(0, sigma_max^2 * I):
            log p_T(z) = -0.5 * ||z||^2 / sigma_max^2
                         - 0.5 * D * log(2 * pi * sigma_max^2)

        where D is the total number of dimensions.
        """
        shape = z.shape
        N = np.prod(shape[1:])  # dimensions excluding batch
        return (
            -0.5 * N * np.log(2 * np.pi * self.sigma_max ** 2)
            - 0.5 * torch.sum(z ** 2, dim=(1, 2, 3)) / (self.sigma_max ** 2)
        )