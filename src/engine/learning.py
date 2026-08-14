import copy
import warnings

import torch
from typing import Callable

from models import utils as models_utils
from models import networks
from . import vmap 
from . import utils as utils


def squared_error_loss(prediction, target, model):
    return torch.sum((target - prediction.reshape_as(target))**2)

def mse_loss(prediction, target, model):
    return torch.mean((target - prediction.reshape_as(target))**2)

def mse_loss_readout_l2(prediction, target, model):
    loss_1 = mse_loss(prediction, target, model)
    loss_2 = torch.sum(model.layers['acc_drive'].weight**2)
    return loss_1 + 2 * loss_2

def learn(
    model, 
    task, 
    loss_fn, 
    num_epochs, 
    optim, 
    svd_by_subpop, 
    device, 
    dynamic_hook_dict=None, 
    save_alphas_and_betas=False,
    save_state_dict=False, 
    save_outputs=False,
    print_update_every_n_epochs=1,
):

    # Could add validation for dynamic_hooks here.

    task.domain = task.domain.to(device)
    task.context_values = task.context_values.to(device)
    task.target = task.target.to(device)
    model.to(device)
    model.train()
    if svd_by_subpop is not None:
        svd_by_subpop = [items.to(device) for items in svd_by_subpop] 

    has_subpopulations = isinstance(
        model.layers['ofc'].gates.subpopulation_membership, 
        networks.SubpopulationMembership
    )

    # Input layer size.
    input_layer_size = model.layers['ofc'].linear.weight.shape[1]

    # Get subpopulation sizes.
    if has_subpopulations:
        subpopulation_sizes = torch.sum(
            model.layers['ofc'].gates.subpopulation_membership.membership, 
            dim=0, 
            dtype=torch.int32
        ) # XXX Will fail if membership matrix is not Boolean, need to add Boolean check
        num_subpops = len(subpopulation_sizes)

        # Preallocate for projections onto modes.
        if save_alphas_and_betas:
            alphas = [
                torch.zeros((num_epochs, subpop_size, input_layer_size))
                for subpop_size in subpopulation_sizes
            ]
            betas = [
                torch.zeros((num_epochs, model.layers['acc_drive'].weight.shape[0], subpop_size))
                for subpop_size in subpopulation_sizes
            ]
        else:
            alphas = []
            betas = []

        # Lengths of projections (is there an inner product here?)
        a = [torch.zeros((num_epochs, input_layer_size)) for _ in subpopulation_sizes]
        b = [torch.zeros((num_epochs, model.layers['acc_drive'].weight.shape[0])) for _ in subpopulation_sizes]
    else:
        alphas = []
        betas = []
        a = []
        b = []

    loss_by_epoch = torch.full((num_epochs,), torch.nan)
    accuracy_by_epoch = torch.full((num_epochs,), torch.nan)

    state_dicts = {}
    outputs = {}

    for i_epoch in range(num_epochs):
        # Update dynamic hooks objects.
        if dynamic_hook_dict is not None:
            dynamic_hook_dict.update(epoch_idx=i_epoch, batch_idx=0)

        # Optionally save model state dict.
        if save_state_dict:
            state_dicts[i_epoch] = copy.deepcopy(model.state_dict())

        # Compute loss and backpropagate.
        if i_epoch % print_update_every_n_epochs == 0:
            print(f"Training for epoch {i_epoch}...")
        optim.zero_grad()
        output, _ = model(task.domain)
        loss = loss_fn(output, task.target)
        loss.backward()
        optim.step()

        # Compute accuracy.
        accuracy = torch.sum(torch.eq(output.squeeze() > 0, task.target > 0))/task.__len__()

        # Optionally save output from current epoch.
        if save_outputs:
            outputs[i_epoch] = output.squeeze()

        # Print and store.
        if i_epoch % print_update_every_n_epochs == 0:
            print(f"Current training loss: {loss}.")
            print(f"Current training accuracy: {accuracy}.")
        loss_by_epoch[i_epoch] = loss.item()
        accuracy_by_epoch[i_epoch] = accuracy.item()

        # Compute projections onto SVD modes.
        if has_subpopulations:
            for i_subpop in range(num_subpops):
                alpha = torch.index_select(
                    model.layers['ofc'].linear.weight, 
                    dim=0,
                    index=models_utils.get_subpopulation_ind(model, 'ofc', i_subpop)
                ) @ svd_by_subpop[2][i_subpop, :, :].T
                beta = svd_by_subpop[0][i_subpop, :, :] @ torch.index_select(
                    model.layers['acc_drive'].weight,
                    dim=1,
                    index=models_utils.get_subpopulation_ind(model, 'ofc', i_subpop)
                )
                # alpha = model.layers['ofc'].linear.weight[get_subpopulation_ind(model, 'ofc', i_subpop), :] @ svd_by_subpop[i_subpop][2]
                # beta = svd_by_subpop[i_subpop][0].T @ model.layers['acc_drive'].weight[:, get_subpopulation_ind(model, 'ofc', i_subpop)]
                a[i_subpop][i_epoch] = torch.sqrt(torch.sum(alpha**2, dim=0)) # TODO: Need to think about how many modes there will be and how to store all of them.
                b[i_subpop][i_epoch] = torch.sqrt(torch.sum(beta**2, dim=1))
                if save_alphas_and_betas:
                    alphas[i_subpop][i_epoch, :, :] = alpha
                    betas[i_subpop][i_epoch, :, :] = beta
    
    if dynamic_hook_dict is not None:
        dynamic_hook_dict.remove()
    print("Done.")
    
    return {
        'alpha': alphas, 
        'betas': betas, 
        'a': a, 
        'b': b, 
        'loss_by_epoch': loss_by_epoch, 
        'accuracy_by_epoch': accuracy_by_epoch, 
        'state_dicts': state_dicts, 
        'outputs': outputs
    }


def vmap_learn(
    models_list,
    task,
    loss_fn,
    num_epochs: int, 
    lr: int =1e-3,
    criteria: dict[str, Callable] = {'accuracy': utils.threshold_and_compute_accuracy},
    device=None, 
    clear_models_list_when_finished=True,
    logger=None,
    log_alphas_and_betas=False,
    log_params=False, 
    log_outputs=False,
    print_update_every_n_epochs=1,
    detach_per_epoch=True
):
    models_list = [model.to(device) for model in models_list]
    task.domain = task.domain.to(device)
    task.target = task.target.to(device)

    if logger is None and (log_alphas_and_betas or log_outputs or log_params):
        warnings.warn(
            f"One of log_alphas_and_betas {log_alphas_and_betas}, "
            f"log_outputs {log_outputs}, or log_params {log_params} "
            "is True, but logger is None, so none of these items will be logged."
        )

    learning = vmap.prepare_for_vmap(
        models_list=models_list,
        loss_fn=loss_fn,
        prototype_model=None,
        clear_models_list_when_finished=clear_models_list_when_finished
    )

    for i_epoch in range(num_epochs):
        if i_epoch % print_update_every_n_epochs == 0:
            print(f"Training for epoch {i_epoch} ...")

        (
            models_grads, 
            models_losses, 
            aux
        ) = vmap.vmap_compute_grads_losses_outputs(
            learning['stateless_grad_and_value'],
            learning['params'],
            learning['buffers'],
            task.domain,
            task.target
        )

        learning['params'] = vmap.update_params(
            learning['params'], 
            models_grads=models_grads,
            lr=lr
        )

        if criteria is not None:
            performance = {
                criterion_name: criterion(aux['y_hat'], task.target) # Should make sure these functions handle broadcasting
                for criterion_name, criterion in criteria.items()
            }

        # Could add functinality here for detaching.
        if detach_per_epoch:
            pass
            # Use recursive functionality to access leaf tensors and detach.
            # performance = {
            #     criterion_name : item.detach() 
            #     for criterion_name, item in performance.items()
            # }

        if logger is not None:
            epoch_log = {}
            epoch_log['loss'] = models_losses
            epoch_log['performance'] = performance if criteria is not None else None
            logger.log_epoch(epoch_idx=i_epoch, **epoch_log)

    print("Done.")
            
    return learning
