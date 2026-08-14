from typing import Callable

import torch
from torch import nn


def build_default_subpopulation_membership(num_neurons, num_subpopulations, device=None):
    """ 
    """
    if num_neurons % num_subpopulations != 0:
        raise ValueError(
            "If using the `build_default_subpopulation_membership_tensor "
            "function, num_ofc_neurons mod num_subpopulations must be 0 but got "
            f"{num_neurons}/{num_subpopulations}={num_neurons%num_subpopulations}."
        )
    subpopulation_membership = torch.zeros((num_neurons, num_subpopulations), dtype=torch.float32, device=device)
    num_neurons_per_subpopulation = num_neurons // num_subpopulations
    for i_subpopulation in range(num_subpopulations):
        subpopulation_membership[
            range(
                i_subpopulation*num_neurons_per_subpopulation, 
                (i_subpopulation+1)*num_neurons_per_subpopulation
            ),
            i_subpopulation
        ] = 1.

    return subpopulation_membership

def build_default_subpopulations_gates_tensor(
    num_subpopulations: int,
    init_value: float | None = None,
    scale: float = 1.,
    softmax: bool = False,
    device=None
):
    gates_tensor = torch.eye(num_subpopulations, device=device)
    num_elements = num_subpopulations**2
    main_diagonal_ind = range(0, num_elements, num_subpopulations+1)
    off_diagonal_ind = torch.tensor(list(set(range(num_elements)).difference(main_diagonal_ind)))
    gates_tensor.put_(
        off_diagonal_ind, 
        torch.tile(
            torch.tensor(
                1/num_subpopulations if init_value is None
                else init_value
            ),
            dims=(off_diagonal_ind.numel(),)
        )
    )
    gates_tensor.mul_(scale)
    if softmax:
        gates_tensor = nn.functional.softmax(gates_tensor, dim=1)

    return gates_tensor
    
def build_default_neurons_gates_tensor(
    num_neurons: int, 
    num_contexts: int,
    init_value: float, 
    scale: float = 1., 
    softmax: bool = False,
    device=None
):
    gates_tensor = torch.tile(torch.tensor(init_value, device=device), dims=(num_contexts, num_neurons)).mul_(scale)
    if softmax:
        gates_tensor = nn.functional.softmax(gates_tensor, dim=0)
    return gates_tensor
    
def build_model_auxiliaries_tensors(
    num_neurons: int,
    num_subpopulations: int | bool,
    num_contexts: int, # Currently only used by gates_builder when there are no subpopulations, which is probably not a good design; however, this module will probably be rewritten after initial tests are passed and vmap is implemented, so may be ok as a stopgap measure
    subpopulation_membership_builder: Callable | None = None,
    gates_builder: Callable | None = None,
    device=None,
    additional_membership_kwargs: dict | None = None,
    additional_gates_kwargs: dict | None = None
):
    
    # Replace sentinel default None with empty dict if necessary.
    additional_gates_kwargs = (
        {} if additional_gates_kwargs is None 
        else additional_gates_kwargs
    )
    
    if num_subpopulations:
        # Replace sentinel default None with empty dict if necessary.
        additional_membership_kwargs = (
            {} if additional_membership_kwargs is None
            else additional_membership_kwargs
        )

        # Build subpopulation membership tensor if there are subpopulations.
        subpopulation_membership_builder = (
            build_default_subpopulation_membership if subpopulation_membership_builder is None
            else subpopulation_membership_builder
        )
        subpopulation_membership_tensor = subpopulation_membership_builder(
            num_neurons=num_neurons, 
            num_subpopulations=num_subpopulations,
            device=device,
            **additional_membership_kwargs
        )

        # Build gates tensor over subpopulations.
        gates_builder = (
            build_default_subpopulations_gates_tensor if gates_builder is None 
            else gates_builder
        )
        additional_gates_kwargs = (
            {} if additional_gates_kwargs is None 
            else additional_gates_kwargs
        )
        gates_tensor = gates_builder(num_subpopulations, device=device, **additional_gates_kwargs)
    else:
        # Placeholder.
        subpopulation_membership_builder = None
        subpopulation_membership_tensor = None

        # Build gates tensor over neurons.
        gates_builder = (
            build_default_neurons_gates_tensor if gates_builder is None 
            else gates_builder
        )
        gates_tensor = gates_builder(
            num_neurons, 
            num_contexts=num_contexts, 
            device=device, 
            **additional_gates_kwargs
        )

    return subpopulation_membership_tensor, gates_tensor