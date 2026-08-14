from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    def init_fn(self, module):
        pass


@dataclass(frozen=True)
class LinearInit(ModuleInit):
    module_type: nn.Module = nn.Linear


@dataclass(frozen=True)
class LinearWeightInitXavierNormal(LinearInit):
    gain: float = 1.
    generator: torch.Generator | None = None

    def init_fn(self, module):
        torch.nn.init.xavier_normal_(module.weight, gain=self.gain, generator=self.generator)


@dataclass(frozen=True)
class LinearWeightInitOrthogonal(LinearInit):
    gain: float = 1.
    generator: torch.Generator | None = None

    def init_fn(self, module):
        torch.nn.init.orthogonal_(module.weight, gain=self.gain, generator=self.generator)


@dataclass(frozen=True)
class LinearWeightInitGaussian(LinearInit):
    mean: float = 0
    std: float = 1.
    generator: torch.Generator | None = None

    def init_fn(self, module):
        torch.nn.init.normal_(module.weight, mean=self.mean, std=self.std, generator=self.generator)
        

@dataclass(frozen=True)
class LinearWeightInitGain(LinearInit):
    gain: float = 1.

    def init_fn(self, module):
        module.weight.mul_(self.gain)
        