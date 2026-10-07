from types import MappingProxyType
from typing import Callable, Literal

import torch

from sci_utils.ml import (
    callbacks, 
    environment, 
    evaluation, 
    hooks, 
    logging,
    loss, 
    metrics, 
    monitoring, 
    rules
)


# Move to engine utils.
DEFAULT_METRIC: metrics.Metric = MappingProxyType({'threshold_accuracy': (metrics.ThresholdAccuracy(threshold=0.))})

def learn(
    model: torch.nn.Module,
    loss_fn: loss.LossFn,
    num_epochs: int,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_val: torch.utils.data.DataLoader | None = None,
    backward_rule: rules.BackwardRule = rules.BackwardI(),
    update_rule: rules.UpdateRule = rules.GradientSystemI(),
    device: str | None = None,
    train_batch_callbacks: list[callbacks.BatchCallback] | None = None,
    train_epoch_callbacks: list[callbacks.EpochCallback] | None = None,
    val_batch_callbacks: list[callbacks.BatchCallback] | None = None,
    val_epoch_callbacks: list[callbacks.EpochCallback] | None = None,
    dynamic_hook_dict: hooks.DynamicHookDict | None = None,
    print_every_n_epochs: int = 50,
):
    """ 
    """
    model.train()
    dataloader_val = dataloader_val or []

    for i_epoch in range(num_epochs):
        # Optionally print updates for this epoch.
        print_this_epoch = i_epoch % print_every_n_epochs == 0
        if print_this_epoch:
            print(f"Training for epoch {i_epoch}...")

        for i_batch, (x, y, batch_aux) in enumerate(dataloader_train):
            # XXX: Hooks will probably fire again before being updated during validation pass.
            if dynamic_hook_dict is not None:
                dynamic_hook_dict.update(epoch_idx=i_epoch, batch_idx=i_batch)

            x = x.to(device) if device is not None else x
            y = y.to(device) if device is not None else y

            train_batch_env = environment.BatchEnvironment(
                epoch_idx=i_epoch,
                batch_idx=i_batch,
                batch_size=x.shape[0],
                x=x,
                y=y
            )
            train_batch_env = evaluation.evaluate(model, train_batch_env, loss_fn)
            params=dict(model.named_parameters())
            backward_rule(train_batch_env.loss, params=params)
            update_rule(params)

            if train_batch_callbacks is not None:
                for callback in train_batch_callbacks:
                    callback(train_batch_env)

        if train_epoch_callbacks is not None:
            for callback in train_epoch_callbacks:
                try:
                    callback(i_epoch)
                except monitoring.StopTraining:
                    return None
            
        # Evaluate on validation set.
        for i_batch, (x, y, batch_aux) in enumerate(dataloader_val):
            val_batch_env = environment.BatchEnvironment(
                epoch_idx=i_epoch,
                batch_idx=i_batch,
                batch_size=x.shape[0],
                x=x,
                y=y
            )
            with evaluation.temporarily_eval(model):
                val_batch_env = evaluation.evaluate(model, val_batch_env, loss_fn)

            if val_batch_callbacks is not None:
                for callback in val_batch_callbacks:
                    callback(val_batch_env)

        if val_epoch_callbacks is not None:
            for callback in val_epoch_callbacks:
                try:
                    callback(i_epoch)
                except monitoring.StopTraining:
                    return None

        if print_this_epoch:
            pass

    return None

def learn_legacy(
    model: torch.nn.Module | list[torch.nn.Module],
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    criteria: dict[str, Callable] = None,
    num_epochs: int = 10,
    lr: float = -1e3,
    svd_by_subpop: torch.Tensor | None = None,
    device: str | None = None,
    print_every_n_epochs: int = 50,
    logger: logging.Logger | None = None,
    log_every_n: int = 1,
    log_gates: str | bool = False,
    noisy_input: torch.distributions.Distribution | None = None
):
    """ 
    Legacy learn function from notebook gamma02 capable of training gated model
    and storing gates. Will look to support training of gated models in main
    learn function if relevant.
    """
    if device is not None:
        model = model.to(device)
        x = x.to(device)
        y = y.to(device)

    if logger is None:
        logger = logging.Logger(log_name='default_logger', verbose_epoch=False)

    for i_epoch in range(num_epochs):
        print_this_epoch = i_epoch % print_every_n_epochs == 0
        if print_this_epoch:
            print(f"Training for epoch {i_epoch}...")

        # Forward and backward passes.
        if noisy_input is not None:
            x_ = x + torch.cat((torch.zeros(x.shape[0], 2, device=device), noisy_input.sample(x[:, 2:].shape)), dim=1)
        else:
            x_ = x
        y_hat, activations = model(x_)
        loss = loss_fn(y_hat, y, model)
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p -= p.grad * lr
                    p.grad = None

        if print_this_epoch:
            print(f"Training loss for epoch {i_epoch}: {loss}.")

        # Compute performance criteria.
        performance = {}
        for name, criterion in criteria.items():
            performance[name] = criterion(y_hat, y)
            if print_this_epoch:
                print(f"{name.capitalize()} for epoch {i_epoch}: {performance[name]['accuracy']}.") # XXX: Hacky for meeting with Thibaud

        # Compute SVD stuff.
        if svd_by_subpop is not None:
            pass

        # Log per epoch items.
        if i_epoch % log_every_n == 0:
            epoch_log = {
                'loss': loss,
                # 'performance': performance,
                # 'activations': activations,
                'accuracy': performance['accuracy'],
                'readout_weights': model.layers['acc_drive'].weight.detach().cpu().clone(),
                'gates': (
                    model.layers[log_gates].gates.get_gate_pattern(x) if log_gates 
                    else None
                )
            }
            epoch_log.update(**performance['accuracy'])
            epoch_log.update(**activations)
            if svd_by_subpop is not None:
                epoch_log.update()
            logger.log_epoch(epoch_idx=i_epoch, **epoch_log)

    return logger

