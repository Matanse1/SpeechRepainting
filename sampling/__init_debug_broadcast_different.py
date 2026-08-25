"""Compatibility import for the canonical predictor-corrector sampler.

Broadcasting is dimension-agnostic in :mod:`sampling.predictors` and
:mod:`sampling.correctors`, so this module intentionally contains no duplicate
sampler, predictor, or corrector implementation.
"""

from . import get_pc_sampler

__all__ = ["get_pc_sampler"]
