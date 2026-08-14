import torch


def apply_threshold(x, threshold, return_dtype: torch.dtype | None = None):
    y = x > threshold
    return y if return_dtype is None else y.to(return_dtype)
    

def compute_accuracy(y_hat, y):

    # Replace with validation logic from tensor module.
    if len(y_hat) != len(y):
        raise ValueError(
            f"`y_hat` and `y` must be the same length but got lengths {len(y_hat)} and {len(y)}, respectively."
        )
    num_elem = len(y)

    # if y_hat.device != y.device:
    #     raise RuntimeError(
    #         f"y_hat and y should be on the same device but got f{y_hat.device} "
    #         f"and {y.device}, respectively."
    #     )

    correct_ind = torch.where(y_hat.reshape_as(y) == y)[0]
    incorrect_ind = torch.arange(
        num_elem,
        device=correct_ind.device
    )[~torch.isin(torch.arange(num_elem, device=correct_ind.device), correct_ind)]
    num_correct = len(correct_ind)
    num_incorrect = len(incorrect_ind)
    accuracy = num_correct / num_elem

    out = {}
    out['num_elem'] = num_elem
    out['correct_ind'] = correct_ind
    out['incorrect_ind'] = incorrect_ind
    out['num_correct'] = num_correct
    out['num_incorrect'] = num_incorrect
    out['accuracy'] = accuracy

    return out

def threshold_and_compute_accuracy(y_hat, y, threshold: float = 0.):
    """ 
    For two float-valued tensors (prediction y_hat and target y), get binary
    values from thresholding, then use these values to compute accuracy.
    """
    y_hat_thresh = apply_threshold(y_hat, threshold=threshold, return_dtype=y_hat.dtype)
    y_thresh = apply_threshold(y, threshold=threshold, return_dtype=y.dtype)
    return compute_accuracy(y_hat_thresh, y_thresh)
