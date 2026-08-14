from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, TYPE_CHECKING
import warnings

import torch
from torch import nn

if TYPE_CHECKING:
    from . import param_init


class SubpopulationMembership(nn.Module):
    """ 
    Callable helper class for networks that have an explicit concept of
    subpopulation membership. Input consists of per subpopulation gate values,
    output consists of gate values for all neurons (neurons can belong to mroe
    than one subpopulation).
    """
    def __init__(
        self,
        membership: torch.Tensor, # num_neurons x num_subpopulations
        nonlinearity: nn.Module = nn.Identity, 
        learn_membership: bool = False
    ):
        super().__init__()
        self.membership = nn.Parameter(membership, requires_grad=learn_membership)
        self.nonlinearity = nonlinearity()
        self.num_neurons_total, self.num_subpopulations = self.membership.shape

    def forward(self, gates):
        return gates @ self.nonlinearity(self.membership).T

    def check_num_neurons_total(self, num_neurons_total):
        """ 
        `num_neurons_total` should be attribute of the encapsulating network.
        Attribute self.num_neurons_total is checked for agreement.
        """
        if self.num_neurons_total != num_neurons_total:
            raise ValueError(
                f"Membership object has self.num_neurons_total as {self.num_neurons_total}, "
                "but `num_neurons_total` (passed in from the encapsulating network) "
                f"is {num_neurons_total}."
            )
        
    def check_num_subpopulations(self, num_subpopulations):
        """ 
        `num_subpopulations` should be attribute of the encapsulating network.
        Attribute self.num_subpopulations is checked for agreement.
        """
        if self.num_subpopulations != num_subpopulations:
            raise ValueError(
                f"Membership object has self.num_subpopulations as {self.num_subpopulations}, "
                "but `num_subpopulations` (passed in from the encapsulating network) "
                f"is {num_subpopulations}."
            )
        
    def get_subpopulation_ind(self, subpopulation_idx):
        if self.membership is None:
            raise ValueError(
                "Subpopulation indices requested, but self.membership is None, " 
                "meaning there is no explicit concept of subpopulation membership."
            )
        if self.membership.requires_grad:
            warnings.warn(
                "Subpopulation indices requested, but self.membership.requires_grad=True "
                "so entries may not be binary. If the i^th j^th element of " 
                "self.membership is nonzero, neuron i will be counted as a " 
                "member of subpopulation j."
            )
        return torch.where(self.membership[:, subpopulation_idx])[0]


class Gates(ABC, nn.Module):
    """
    This base class captures the fact that regardless of how gates are
    retrieved/computed, the gated network may or may not have an explicit
    concept of subpopulation membership.
    """
    def __init__(self, subpopulation_membership: SubpopulationMembership | None = None):
        super().__init__()
        self.subpopulation_membership = subpopulation_membership

    @abstractmethod
    def get_gate_pattern(self, x): 
        pass

    def forward(self, x):
        """ 
        Dispatches to subclass specific routine to get gate pattern, then
        handles common potential interception by SubpopulationMembership
        object, which handles the case where gate values correspond to
        subpopulations; with an appropriate method of combination for neurons
        in multiple subpopulations, it broadcasts subpopulation specific gate
        values to all neurons. If there is no membership object, gates are
        treated as per neuron.
        """
        gate_pattern = self.get_gate_pattern(x)
        return (
            self.subpopulation_membership(gate_pattern) if self.subpopulation_membership is not None 
            else gate_pattern
        )
        

class GatesFromOracle(Gates):
    """ 
    """
    def __init__(
        self, 
        input_ind: Sequence[int], # Specifies which indices along final dim correspond to oracle input
        all_gate_patterns: torch.Tensor,
        learn_gates: bool = False,
        nonlinearity: nn.Module = nn.Identity,
        subpopulation_membership: SubpopulationMembership | None = None, # If membership is None, gates are assumed to apply to neurons, not subpopulations
    ):  
        super().__init__(subpopulation_membership=subpopulation_membership)
        self.input_ind = input_ind
        self.all_gate_patterns = nn.Parameter(all_gate_patterns, requires_grad=learn_gates)
        self.nonlinearity = nonlinearity() if isinstance(nonlinearity, type) else nonlinearity

    def get_gate_pattern(self, x):
        """ 
        Accepts a batch_size x num_features input and returns a batch_size x num_gates tensor.
        """
        gate_pattern_to_use_one_hot_ind = torch.nonzero(x[..., self.input_ind], as_tuple=True)[1]
        return self.nonlinearity(self.all_gate_patterns[gate_pattern_to_use_one_hot_ind, :])
        

class GatesNetwork(Gates):
    """ 
    TODO: if input_ind is not None, validate length against in_features.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        nonlinearity: nn.Module = nn.Sigmoid,
        input_ind: Sequence[int] | None = None, # Can give as input to gate network just a subset of the full input
        subpopulation_membership: SubpopulationMembership | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        # store_last_gate_pattern: bool = False # When training in a typical stateful setup, can intercept gates and read off in training loop. An idea: if we're recomputing in vmap anyway, maybe just recompute here as well.
    ):
        super().__init__(subpopulation_membership=subpopulation_membership)
        self.in_features = in_features
        self.out_features = out_features
        self.input_ind = input_ind
        self.linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.nonlinearity = nonlinearity() if isinstance(nonlinearity, type) else nonlinearity

        # self._store_last_gate_pattern = store_last_gate_pattern
        # self.last_gate_pattern = None
        
    def get_gate_pattern(self, x):
        """ 
        Accepts as input a batch_size a num_features tensor and returns a batch_size x num_gates tensor.
        """
        x_input = x[..., self.input_ind] if self.input_ind is not None else x 
        gate_pattern = self.nonlinearity(self.linear(x_input))

        # if self._store_last_gate_pattern:
        #     self.last_gate_pattern = gate_pattern

        return gate_pattern
    

class GatedLayer(nn.Module):
    """ 
    TODO: Add an input_ind argument so that the linear layer can also accept a subset of input dims.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        gates: Gates,
        bias: bool = False,
        nonlinearity: nn.Module = nn.Identity,
        input_ind: Sequence[int] | None = None, # This applies only to the input given to the linear layer, not to the gates object (which is responsible for deciding for itself which part of the input to use)
        init_fns: Sequence[param_init.ModuleInit] | None = None,
        device: str | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.input_ind = input_ind
        self.linear = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.nonlinearity = nonlinearity() if isinstance(nonlinearity, type) else nonlinearity
        self.gates = gates

        if init_fns is not None:
            for fn in init_fns:
                self.apply(fn)

    def forward(self, x):
        x_input = x[..., self.input_ind] if self.input_ind is not None else x
        return self.nonlinearity(self.linear(x_input)) * self.gates(x) # If self.gates(x) is 1d, will be treated as row vector, as desired (in case output of self.linear(x) is square)


class NamedSequential(nn.Module):
    """ 
    """
    def __init__(self, layers: dict[str, nn.Module], init_fns=None):
        """ 
        Order of layers is order of insertion of key-value pairs in dictionary.
        """
        super().__init__()
        self.layers = nn.ModuleDict(layers)

        if init_fns is not None:
            for fn in init_fns:
                self.apply(fn)

    def forward(self, x, return_activations: list[str] | None = None):
        """ 
        """
        return_activations = (
            set(return_activations) 
            if return_activations is not None else set()
        )
        activations = {}
        for name, layer in self.layers.items():
            y = layer(x)
            if name in return_activations:
                activations[name] = y
            x = y

        return x, activations
                