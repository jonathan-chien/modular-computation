import itertools

import torch


def cart_prod_integers(m, n, return_as: str = 'list'):
    """ 
    Returns n-fold Cartesian product of the set of integers {1, 2, ..., m}.
    Default is to return product as list, but tuple or itertools.product object
    itself can also be returned.
    """
    base = [[i for i in range(m)] for ii in range(n)]
    prod = itertools.product(*base)
    if return_as == 'list':
        return list(prod)
    elif return_as == 'tuple':
        return tuple(prod)
    elif return_as == 'itertools.product':
        return prod
    else:
        raise ValueError(f"Unrecognized value '{return_as}' for `return_as`. Must be 'list', 'tuple', or 'itertools.product'.")

def cart_prod_one_hot(encoding, num_vars_per_set, concatenate=False):
    """ 
    Consider an integer M_i. Represent A_1 = {1, 2, ..., M_i} in "one hot"
    form, that is as a set of M_i strings where the k_th string has a 1 at the
    k_th bit and 0's elsewhere. Now represent these strings as lists of bits,
    and take the Cartesian product of A_1, A_2, ..., A_n. Input is
    num_items_per_set = [M_1, M_2, ..., M_n]. Optionally, the n lists in each
    element of the product can be concatenated to yield one list of length m*n
    (this is not done by default).
    """
    base = [
        [
            [
                encoding[0] if i_bit != i_val else encoding[1] 
                for i_bit in range(num_vars_per_set[i_set])
            ]
            for i_val in range(num_vars_per_set[i_set])
        ]
        for i_set in range(len(num_vars_per_set))
    ]
    list_of_tuples_of_lists = list(itertools.product(*base))
    out = list_of_tuples_of_lists

    if concatenate:
        list_of_lists = [
            list(itertools.chain.from_iterable(tuple_)) 
            for tuple_ in list_of_tuples_of_lists
        ]
        out = list_of_lists

    return out

def create_dict_from_key_and_value_lists(key_list, value_list):
    """ 
    Enforces matching lengths of key list and value list and fills in integer
    indices as keys if no key list is provided.
    """
    if key_list is None:
        d = {idx : value_list[idx] for idx in range(len(value_list))}
    else:
        if len(key_list) != len(value_list):
            raise ValueError(
                "key_list and value_list must be the same length but got "
                f"lengths of {len(key_list)} and {len(value_list)}."
            )
        else:
            d = dict(zip(key_list, value_list))
    return d

def compute_cross_covariance_operator(input: torch.Tensor, output: torch.Tensor):
    # `input`` is an m x n tensor with m = num_samples and n = num_input_features.
    # `output`` is either a 1D tensor of length m (for 1D output) or an m x p 
    # tensor with p = num_output_features.
    
    # Validate input and prepare for broadcasting.
    if len(input.shape) != 2:
        raise ValueError(
            f"task.domain should be 2D but got a {len(input.shape)} tensor."
        )
    input = input[None, :, :]

    # Validate output and prepare for broadcasting.
    if len(output.shape) == 1:
        output = output[:, None]
    elif len(output.shape) != 2:
        raise ValueError(f"task.target should be 1D or 2D but got {len(output.shape)}D tensor.")
    output = output.T
    output = output[:, :, None]

    return torch.sum(output * input, dim=1).squeeze()
