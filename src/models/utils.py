from __future__ import annotations
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .networks import NamedSequential


# Helper methods for when working with a GatedLayer ofc layer and a nn.Linear acc layer.
def get_subpopulation_ind(model: NamedSequential, layer_name: str, subpopulation_idx: int):
    """ 
    Helper method to access subpopulation indices when working with a GatedLayer 
    that is a module of a SequentialNamed network.
    """
    return model.layers[layer_name].gates.subpopulation_membership.get_subpopulation_ind(subpopulation_idx)

def get_subpopulation_view(model: NamedSequential, layer_name: str, subpopulation_idx: int):
    """ 
    Helper method to access subpopulation view when working with a GatedLayer 
    that is a module of a SequentialNamed network.
    """
    subpopulation_ind = get_subpopulation_ind(model, layer_name, subpopulation_idx)
    return model.layers[layer_name].linear.weight[subpopulation_ind, :]

def compute_cross_cov_operator_by_subpop(
    context_specific_cross_cov_operators: torch.Tensor, 
    gates_matrix: torch.Tensor, # num_contexts x num_subpopulations
    compute_svd: bool = True
):
    """ 
    Each operator's matrix representation should have the same shape.
    """
    num_contexts, num_subpopulations = gates_matrix.shape
    operator_shape = context_specific_cross_cov_operators.shape[1:]

    cross_cov_operator_by_subpopulation = torch.empty(
        size=(num_subpopulations,) + operator_shape
    )
    for i_subpopulation in range(num_subpopulations):
        for i_context in range(num_contexts):
            cross_cov_operator_by_subpopulation[i_context, :, :] = (
                context_specific_cross_cov_operators[i_context, :, :] 
                * gates_matrix[i_context, i_subpopulation]
            )
    
    u_by_subpopulation = torch.empty(num_contexts, operator_shape[0], operator_shape[0])
    s_by_subpopulation = torch.empty(num_contexts, min(operator_shape))
    v_by_subpopulation = torch.empty(num_contexts, operator_shape[1], operator_shape[1])
    if compute_svd:
        for i_subpopulation in range(num_subpopulations):
            (
                u_by_subpopulation[i_subpopulation, :, :], 
                s_by_subpopulation[i_subpopulation, :], 
                v_by_subpopulation[i_subpopulation, :, :] 
            ) = torch.linalg.svd(
                cross_cov_operator_by_subpopulation[i_subpopulation, :, :],
                full_matrices=True
            )

    return cross_cov_operator_by_subpopulation, (u_by_subpopulation, s_by_subpopulation, v_by_subpopulation)
