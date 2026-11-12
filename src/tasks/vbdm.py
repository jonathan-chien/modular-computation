from abc import ABC, abstractmethod
from typing import Callable
import re

import torch

from . import dm


class ContextDependentValueBased2AFC(dm.DecisionMaking):
    def __init__(
        self,
        stimuli: list[dm.Stimulus],
        stimulus_to_value_map: Callable = lambda x: x,
        valid_task_cond_function: Callable | None = None,
        c1: float = 1.,
        c2: float = 1.,
        c3: float = 1.,
        c4: float = 1.,
        c5: float = 0.,
        c6: float = 0.,
        s1: float = 1.,
        s2: float = 1.,
        s3: float = 1.,
        s4: float = 1.,
        s5: float = 0.,
        s6: float = 0.,
        cs: float = 1.,
        zscore_domain: bool = False,
        transform=None
    ):
        self.stimulus_to_value_map = stimulus_to_value_map
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        self.c5 = c5
        self.c6 = c6
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.s5 = s5
        self.s6 = s6
        self.cs = cs
        super().__init__(
            stimuli=stimuli, 
            valid_task_cond_function=(
                lambda task_cond: self.dv_function(task_cond)[0] != 0. if valid_task_cond_function is None
                else valid_task_cond_function
            ),
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

    def dv_function(self, task_cond: dm.TaskCondition):
        """ 
        """
        color_diff = (
            self.stimulus_to_value_map(task_cond.get(stimulus='option_1', feature='color')) * self.c1
            - self.stimulus_to_value_map(task_cond.get(stimulus='option_2', feature='color')) * self.c2
        ) * self.c3
        shape_diff = (
            self.stimulus_to_value_map(task_cond.get(stimulus='option_1', feature='shape')) * self.s1
            - self.stimulus_to_value_map(task_cond.get(stimulus='option_2', feature='shape')) * self.s2
        ) * self.s3
        color_times_shape_diff = (
            self.stimulus_to_value_map(task_cond.get(stimulus='option_1', feature='color')) * self.c1
            * self.stimulus_to_value_map(task_cond.get(stimulus='option_1', feature='shape')) * self.s1
            - self.stimulus_to_value_map(task_cond.get(stimulus='option_2', feature='color')) * self.c2
            * self.stimulus_to_value_map(task_cond.get(stimulus='option_2', feature='shape')) * self.s2
        ) * self.cs

        color_dv = self.c4 * color_diff + self.c5 * color_diff * shape_diff + self.c6 * color_times_shape_diff
        shape_dv = self.s4 * shape_diff + self.s5 * color_diff * shape_diff + self.s6 * color_times_shape_diff

        if torch.all(task_cond.get(stimulus='context', feature='') == torch.tensor([1, 0], device=color_diff.device)):
            relevant_diff, irrelevant_diff = color_dv, shape_dv
        elif torch.all(task_cond.get(stimulus='context', feature='') == torch.tensor([0, 1], device=shape_diff.device)):
            relevant_diff, irrelevant_diff = shape_dv, color_dv
        else:
            raise ValueError(
                "Unexpected value of " 
                f"{task_cond.get(stimulus='context', feature='')} " 
                "for choice item 0."
            )

        dv = relevant_diff
        aux = {'relevant_diff': relevant_diff, 'irrelevant_diff': irrelevant_diff}
        return dv, aux
