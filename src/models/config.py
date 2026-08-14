from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from .networks import SubpopulationMembership
    from .param_init import ModuleInit


# TODO: Add inheritance from ArgsConfig eventually.

@dataclass(frozen=True)
class SubpopulationMembershipConfig:
    membership: torch.Tensor # num_neurons x num_subpopulations
    membership_nonlinearity: nn.Module = nn.Identity
    learn_membership: bool = False

@dataclass(frozen=True)
class GatesConfig:
    subpopulation_membership: SubpopulationMembership | None

@dataclass(frozen=True)
class GatesNetworkConfig(GatesConfig):
    in_features: int
    out_features: int
    bias: bool = False
    nonlinearity: nn.Module = nn.Sigmoid
    input_ind: Sequence[int] | None = None # Can give as input to gate network just a subset of the full input
    device: str | None = None
    dtype: torch.dtype | None = None

@dataclass(frozen=True)
class GatesFromOracleConfig:
    input_ind: Sequence[int] # Specifies which indices along final dim correspond to oracle input
    all_gate_patterns: torch.Tensor
    learn_gates: bool = False,
    nonlinearity: nn.Module = nn.Identity,
        
@dataclass(frozen=True)
class GatedLayerConfig:
    """ 
    Gates object has separate config and should be injected at instantiation time.
    """
    in_features: int
    out_features: int
    bias: bool = False
    nonlinearity: nn.Module = nn.Identity
    input_ind: Sequence[int] | None = None # This applies only to the input given to the linear layer, not to the gates object (which is responsible for deciding for itself which part of the input to use)
    init_fns: Sequence[ModuleInit] | None = None
    device: str | None = None
    dtype: torch.dtype | None = None