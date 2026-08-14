import copy

import torch


def stateless_forward_closure(prototype_model):
    """ 
    Produces stateless nn.Module forward via closure over prototype model.
    """
    def stateless_forward(params, buffers, x):
        y_hat, activations = torch.func.functional_call(prototype_model, (params, buffers), (x,))
        return y_hat, activations
    return stateless_forward

def stateless_loss_closure(stateless_forward, loss_fn):
    """ 
    Produces stateless loss functional (in the math sense here) via closure
    over a stateless nn.Module forward.
    """
    def stateless_loss(params, buffers, x, y):
        y_hat, activations = stateless_forward(params, buffers, x)
        loss = loss_fn(y_hat, y) # TODO: Look into adding model regularization
        aux = {'y_hat': y_hat, 'activations': activations}
        return loss, aux
    return stateless_loss

def prepare_stateless_forward(models_list: list, prototype_model=None, clear_models_list_when_finished=True):
    """ 
    Accepts a list of nn.Modules (must have the same architecture) and prepares 
    for vmap by extracting parameters, buffers, and preparing a stateless forward.
    """
    params, buffers = torch.func.stack_module_state(models_list)
    prototype_model = (
        copy.deepcopy(models_list[0]).to('meta') if prototype_model is None 
        else prototype_model
    )
    stateless_forward = stateless_forward_closure(prototype_model)

    if clear_models_list_when_finished:
        models_list.clear()

    return stateless_forward, params, buffers, prototype_model

def prepare_for_vmap(models_list, loss_fn, prototype_model=None, clear_models_list_when_finished=True):
    stateless_forward, params, buffers, prototype_model = prepare_stateless_forward(
        models_list,
        prototype_model=prototype_model,
        clear_models_list_when_finished=clear_models_list_when_finished
    )
    stateless_loss = stateless_loss_closure(stateless_forward, loss_fn)
    stateless_grad_and_value = torch.func.grad_and_value(stateless_loss, argnums=0, has_aux=True)

    return {
        'stateless_grad_and_value': stateless_grad_and_value, 
        'stateless_forward': stateless_forward, 
        'stateless_loss': stateless_loss, 
        'params': params, 
        'buffers': buffers, 
        'prototype_model': prototype_model
    }

def vmap_compute_grads_losses_outputs(stateless_grad_and_value, params, buffers, x, y):
    models_grads, (models_losses, aux) = torch.func.vmap(
        stateless_grad_and_value, 
        in_dims=(0, 0, None, None)
    )(params, buffers, x, y)
    return models_grads, models_losses, aux

def update_params(params, models_grads, lr):
    return {
        name: p - lr * models_grads[name]
        for name, p in params.items()
    }

def vmap_forward(stateless_forward, params, buffers, x):
    return torch.func.vmap(
        stateless_forward, in_dims=(0, 0, None)
    )(params, buffers, x)

# def vmap_forward(models_list: list, x, prototype_model=None):
#     """ 
#     Prepares params, buffers, and stateless version of model, then hands to vmap.
#     """
#     # params, buffers = torch.func.stack_module_state(models_list)
#     # prototype_model = models_list[0] if prototype_model is None else prototype_model
#     # stateless_forward = get_stateless_forward(prototype_model)
#     stateless_forward, params, buffers = prepare_stateless_forward(models_list, prototype_model)
#     y_hat = torch.func.vmap(stateless_forward, in_dims=(0, 0, None))(params, buffers, x)
    
#     return y_hat, params, buffers

# def vmap_compute_grads_and_losses(params, buffers, x, y, stateless_forward, loss_fn):
#     stateless_loss = stateless_loss_closure(stateless_forward, loss_fn)
#     stateless_grad_and_value = torch.func.grad_and_value(stateless_loss, argnums=0)
#     models_grads, models_losses = torch.vmap(
#         stateless_grad_and_value, 
#         in_dims=(0, 0, None, None)
#     )(params, buffers, x, y)
    
#     return models_grads, models_losses, params, buffers

  

# def vmap_compute_losses(models_list, x, y, loss_fn, prototype_model=None):
#     stateless_forward, params, buffers = prepare_stateless_forward(models_list, prototype_model)
#     stateless_loss = stateless_loss_closure(stateless_forward, loss_fn)
#     models_losses = torch.func.vmap(stateless_loss, in_dims=(0, 0, None, None))(params, buffers, x, y)

#     return models_losses, params, buffers


# def compute_loss(models_list, x, y, loss_fn, prototype_model=None):
#     stateless_forward, params, buffers = prepare_stateless_forward(models_list, prototype_model)
#     y_hat = stateless_forward(params, buffers, x)
#     return loss_fn(y_hat, y)

# def vmap_compute_loss(models_list, x, y, loss_fn, prototype_model=None):
#     torch.func.vmap(compute_loss, input_dims=(0))
