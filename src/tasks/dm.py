from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import itertools
import re
from typing import Callable, Sequence

import torch


@dataclass(frozen=False)
class Stimulus:
    """ 
    Using torch.Tensor as input data type because it more natively handles 1d 
    vs 2d tensors (using tuples/lists is more unwieldy).
    """
    # Initialization parameters.
    stimulus_name: str
    features: dict[str, torch.Tensor] # dict with feature names as keys pointing to Sequence of tuples of floats (even scalar floats should be wrapped in tuple)
    feature_names: list[str] = field(init=False)

    # Derived variables.
    combinations: dict[str, object] = field(init=False)
    num_combinations: int = field(init=False)

    def __post_init__(self):
        """ 
        """
        self.feature_names = list(self.features.keys())
        cart_prod = list(itertools.product(*self.features.values()))
        cart_prod = [[torch.atleast_1d(tensor) for tensor in element] for element in cart_prod]
        self.combinations = [
            dict(zip(self.feature_names, element)) for element in cart_prod
        ]
        self.num_combinations = len(self.combinations)


@dataclass(frozen=False)
class TaskCondition:
    """ 
    A task condition defined by a unique set of values across choice items
    (e.g. context, option1, option 2).
    """
    stimuli: dict[str, dict[str, float]]
    total_condition_idx: int
    valid_condition_idx: int = field(default=None, init=False)
    dv: float = field(default=None, init=False)
    aux: dict[str, object] = field(default=None, init=False)

    def get(self, stimulus: str, feature: str):
        return self.stimuli[stimulus][feature]

    def compute_dv(self, dv_function):
        self.dv, self.aux = dv_function(self)
        return self.dv, self.aux

    def concatenate(self):
        return torch.cat(
            tuple(
                torch.cat(
                    tuple(stimulus.values())
                ) for stimulus in self.stimuli.values()
            )
        )


class DecisionMaking(ABC, torch.utils.data.Dataset):
    def __init__(
            self, 
            stimuli: list[Stimulus], 
            valid_task_cond_function: Callable | None = None,
            zscore_domain: bool = False,
            transform=None
        ):
        # Input parameters.
        self.stimuli = stimuli
        self.valid_task_cond_function = valid_task_cond_function
        self.zscore_domain = zscore_domain
        self.transform = transform

        # Initilization logic.
        super().__init__()
        self.valid_task_conds, self.all_task_conds = self._initialize_task_conditions()
        self.dv = self.compute_dv(self.valid_task_conds)
        self.domain = self._initialize_domain(self.valid_task_conds, self.zscore_domain)
               
    def _initialize_task_conditions(self):
        """ 
        """
        stimuli_names = [stimulus.stimulus_name for stimulus in self.stimuli]
        
        cart_prod = list(
            itertools.product(
                *[stimulus.combinations for stimulus in self.stimuli]
            )
        )

        all_task_conds = [
            TaskCondition(
                total_condition_idx=i_element, stimuli=dict(zip(stimuli_names, element))
            )
            for i_element, element in enumerate(cart_prod)
        ]
        
        # Identify index of invalid choices.
        ind_to_remove = []
        for i_task_cond, task_cond in enumerate(all_task_conds):
            if not self.is_valid_task_cond(task_cond):
                ind_to_remove.append(i_task_cond)
        
        # Reconstitute lists corresponding to valid choices.
        valid_task_conds = []
        valid_idx = 0
        for i_task_cond, task_cond in enumerate(all_task_conds):
            if i_task_cond not in ind_to_remove:
                task_cond.valid_condition_idx = valid_idx
                valid_task_conds.append(task_cond)
                valid_idx += 1
            
        return valid_task_conds, all_task_conds

    def _initialize_domain(
        self, 
        valid_task_conds: list[TaskCondition], 
        zscore_domain: bool
    ): 
        domain = torch.cat(
            [
                task_cond.concatenate().unsqueeze(0) 
                for task_cond in valid_task_conds
            ], 
            dim=0
        )
        if zscore_domain:
            domain = self._zscore_domain(domain)
        return domain

    def compute_dv(self, valid_task_conds: list[TaskCondition]):
            dv_valid_task_conds = []
            for task_cond in valid_task_conds:
                # Compute decision variable and auxiliary data for current task condition.
                task_cond.compute_dv(self.dv_function)
    
                # Append to growing list that will be converted into tensor and used as "target" during training.
                dv_valid_task_conds.append(task_cond.dv)
    
            return torch.cat(dv_valid_task_conds)

    def is_valid_task_cond(self, task_cond):
        return (
            True if self.valid_task_cond_function is None 
            else self.valid_task_cond_function(task_cond)
        )
    
    def __len__(self):
        return len(self.domain)

    def __getitem__(self, idx):
        data = (
            self.transform(self.domain[idx]) if self.transform is not None 
            else self.domain[idx]
        )
        return data, self.dv[idx], self.valid_task_conds[idx]

    @abstractmethod
    def dv_function(self):
        """ 
        Should return tuple (dv: float, aux: dict[str, object]).
        """
        pass

    @abstractmethod
    def _zscore_domain(zscore_domain):
        pass

    @abstractmethod
    def get_condition_ind(self):
        pass

    @staticmethod
    def collate_fn(batch):
        """ 
        """
        x, y, valid_task_conds = zip(*batch)
        x_cat = torch.stack(x, dim=0)
        y_cat = torch.stack(y, dim=0)
        return x_cat, y_cat, valid_task_conds

            
class ContextDependentAnalogue2AFC(DecisionMaking):
    def __init__(
            self,
            stimuli: list[Stimulus], # Inputs must be analogue representation of value
            valid_task_cond_function: Callable | None = (
                lambda task_cond: ContextDependentAnalogue2AFC.dv_function(task_cond)[0] != 0
            ),
            zscore_domain: bool = False,
            transform=None,
        ):
        super().__init__(
            stimuli=stimuli, 
            valid_task_cond_function=valid_task_cond_function,
            zscore_domain=zscore_domain,
            transform=transform
        )

    def _zscore_domain(self, domain):
            """ 
            """
            domain[:, 2:] = (domain[:, 2:] - torch.mean(domain[:, 2:])) / torch.std(domain[:, 2:])
            return domain

    def get_condition_ind(self, trial_type: str):
        """ 
        """
        if trial_type in ['congruent', 'incongruent']:
            if trial_type == 'congruent':
                def comparison(a): return a > 0
            elif trial_type == 'incongruent':
                def comparison(a): return a < 0
            else:
                raise ValueError(
                    f"Unexpected value '{trial_type}' for trial_type. "
                    "Must be 'congruent' or 'incongruent'."
                )

            return [
                task_cond.condition_idx for task_cond in self.valid_task_conds 
                if comparison(
                    task_cond.aux['relevant_difference'] * task_cond.aux['irrelevant_difference'] 
                )
            ]
        elif trial_type[:-1] == 'context:':
            try:
                context = tuple(int(s) for s in re.sub('context:', '', trial_type))
            except ValueError as e:
                raise ValueError(
                    "Tried to call remaining string following 'context:' in " 
                    f"`trial_type` to tuple of ints and got a ValueError: {e}. "
                    "Ensure final element is string representation of a sequence of integers."
                )
            return [
                task_cond.condition_idx for task_cond in self.valid_task_conds
                if task_cond.get(stimulus_name='context', feature_name='') == context
            ]

    @staticmethod
    def dv_function(task_cond: TaskCondition):
        """ 
        """
        color_diff = (
            task_cond.get(stimulus='option_1', feature='color')
            - task_cond.get(stimulus='option_2', feature='color')
        )
        shape_diff = (
            task_cond.get(stimulus='option_1', feature='shape')
            - task_cond.get(stimulus='option_2', feature='shape')
        )
        if torch.all(task_cond.get(stimulus='context', feature='') == torch.tensor([1, 0], device=color_diff.device)):
            relevant_diff, irrelevant_diff = color_diff, shape_diff
        elif torch.all(task_cond.get(stimulus='context', feature='') == torch.tensor([0, 1], device=shape_diff.device)):
            relevant_diff, irrelevant_diff = shape_diff, color_diff
        else:
            raise ValueError(
                "Unexpected value of " 
                f"{task_cond.get(stimulus='context', feature='')} " 
                "for choice item 0."
            )

        dv = relevant_diff
        aux = {'relevant_diff': relevant_diff, 'irrelevant_diff': irrelevant_diff}
        return dv, aux
        

class ContextPermTransform:
    """
    TODO: Under development as of 2026/08/27.
    """
    def __init__(self, prob: float, context_ind: Sequence[int], perm=None):
        """ 
        """
        self.bernoulli = torch.distributions.Bernoulli(probs=prob)
        self.perm = perm
        self.context_ind = context_ind
        self.num_context_coord = len(self.context_ind)

    def __call__(self, x, rand_perm_generator=None):
        """ 
        """
        num_obs = x.shape[0]
        mask = self.bernoulli.sample(sample_shape=(num_obs,)) == 1.
        permutations = torch.cat(
            [
                (
                    torch.randperm(
                        n=self.num_context_coord, 
                        generator=rand_perm_generator
                    )
                    if self.perm is None else self.perm 
                ) for i_obs in range(num_obs) if mask[i_obs] 
            ]
        )
        view = x[mask][:, self.context_ind]
        view[:] = view[:, permutations]
        