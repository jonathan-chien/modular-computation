from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import torch
from torch import nn


@dataclass(frozen=True)
class ModuleInit(ABC):
    module_type: nn.Module

    def __call__(self, module: nn.Module):
        if isinstance(module, self.module_type):
            with torch.no_grad():
                self.init_fn(module)

    @abstractmethod
    def init_fn(self, module: nn.Module):
        pass


@dataclass(frozen=True)
class LinearInit(ModuleInit):
    module_type: nn.Module = nn.Linear


@dataclass(frozen=True)
class LinearInitWeightXavierNormal(LinearInit):
    gain: float = 1.
    generator: torch.Generator | None = None

    def init_fn(self, module: nn.Module):
        nn.init.xavier_normal_(module.weight, gain=self.gain, generator=self.generator)


@dataclass(frozen=True)
class LinearInitWeightOrthogonal(LinearInit):
    gain: float = 1.
    generator: torch.Generator | None = None

    def init_fn(self, module: nn.Module):
        nn.init.orthogonal_(module.weight, gain=self.gain, generator=self.generator)


@dataclass(frozen=True)
class LinearInitWeightGaussian(LinearInit):
    mean: float = 0
    std: float = 1.
    generator: torch.Generator | None = None

    def init_fn(self, module: nn.Module):
        nn.init.normal_(module.weight, mean=self.mean, std=self.std, generator=self.generator)
        

@dataclass(frozen=True)
class LinearInitWeightGain(LinearInit):
    gain: float = 1.

    def init_fn(self, module: nn.Module):
        module.weight.mul_(self.gain)


@dataclass(frozen=True)
class LinearInitBiasGain(LinearInit):
    gain: float = 0.
    warn_on_none: bool = True

    def init_fn(self, module: nn.Module):
        if module.bias is None:
            if self.warn_on_none:
                warnings.warn(
                    "Attempted to apply gain to module.bias, but module.bias "
                    "is None. No action was taken."
                )
        else:
            module.bias.mul_(self.gain)
