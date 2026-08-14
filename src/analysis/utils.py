import numpy as np
import torch

def pca(x, corr_mat=False):
    """ 
    Batchwise real-valued PCA where leading n-2 dims are treated as batch dims. 
    n=2 corresponds to standard PCA on a single matrix.
    """
    dof = x.shape[-2] - 1
    x = x - np.mean(x, axis=-2, keepdims=True)
    if corr_mat:
        x /= np.std(x, axis=-2, ddof=1, keepdims=True)
    
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    v = vh.swapaxes(-2, -1)

    pcs = u*s[..., np.newaxis, :]
    lambdas = s**2 / dof
    loadings =  v * np.sqrt(lambdas[..., np.newaxis, :])

    return pcs, lambdas, loadings, v
