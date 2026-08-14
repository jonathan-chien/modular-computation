from abc import ABC, abstractmethod
import itertools

import torch

from . import utils


class ChoiceItem:
    def __init__(self, feature_values, feature_names=None, encoding='analogue', representation_dtype=None):
        self.features = utils.create_dict_from_key_and_value_lists(feature_names, feature_values)
        self.encoding = encoding
        self.representation_dtype = representation_dtype

        # Generate all combinations of all feature values.
        self.values = list(itertools.product(*self.features.values()))

        # Generate representation for each combination of feature values.
        if isinstance(self.encoding, str):
            if self.encoding != 'analogue':
                raise ValueError(
                    "If self.encoding is a str, it must have value 'analogue' "
                    f"but got '{self.encoding}'."
                )
            self.representations = torch.tensor(self.values, dtype=self.representation_dtype)
        elif not isinstance(self.encoding, str):

            # TODO: Add validation logic from general_utils here.

            if len(self.encoding) != 2:
                raise ValueError(
                    "If not str, self.encoding should be iterable of "
                    f"length 2 with int/float values, but got type {self.encoding} "
                    f"of length {len(self.encoding)}."
                )
            self.representations = torch.tensor(
                    utils.cart_prod_one_hot(
                        encoding=self.encoding,
                        num_vars_per_set=[
                            len(single_feature_values) 
                            for single_feature_values in self.features.values()
                        ],
                        concatenate=True
                    ),
                dtype=self.representation_dtype
            )

        self.num_states = len(self.values)
        

class ValueBasedDecisionMaking(ABC, torch.utils.data.Dataset):
    def __init__(
            self, 
            choice_items: list[ChoiceItem], 
            choice_names: list[str],
            target_dtype: torch.dtype,
            zscore_domain: bool = False
        ):
        super().__init__()
        self.choice_items = utils.create_dict_from_key_and_value_lists(choice_names, choice_items)
        self.choices = self._initialize_choices()
        self.relevant_differences, self.irrelevant_differences = self._compute_differences(self.choices)
        self.target = torch.tensor(self.relevant_differences, dtype=target_dtype)
        self.zscore_domain = zscore_domain
        self.domain = self._initialize_domain(self.choices, self.zscore_domain)
               
    def _initialize_choices(self):
        # Get all possible choices.
        all_choices = {
            quantity : list(itertools.product(
                *[getattr(choice_item, quantity) for choice_item in self.choice_items.values()]
            ))
            for quantity in ['values', 'representations']
        }
        
        # Identify index of invalid choices.
        ind_to_remove = []
        for i_choice, choice in enumerate(all_choices['values']):
            if not self._valid_choice_condition(choice):
                ind_to_remove.append(i_choice)
        
        # Reconstitute lists corresponding to valid choices.
        valid_choices = {
            quantity : [
                choice for i_choice, choice in enumerate(all_choices[quantity]) 
                if i_choice not in ind_to_remove
            ]
            for quantity in ['values', 'representations']
        }
            
        return valid_choices

    def _compute_differences(self, choices):
        # Compute decision variable for each choice.
        return zip(*[self._target_function(choice) for choice in choices['values']])
 
    def _initialize_domain(self, choices, zscore_domain): 
        domain = torch.cat([torch.cat(choice).unsqueeze(0) for choice in choices['representations']], dim=0)
        if zscore_domain:
            domain = self._zscore_domain(domain)
        return domain
    
    def __len__(self):
        return len(self.domain)

    def __getitem__(self, idx):
        return (
            self.domain[idx], 
            self.relevant_differences[idx], 
            self.irrelevant_differences[idx]
        )
    
    @abstractmethod
    def _valid_choice_condition(self):
        pass

    @abstractmethod
    def _target_function(self):
        pass

    @abstractmethod
    def _zscore_domain(zscore_domain):
        pass

    @abstractmethod
    def get_sample_ind(self):
        pass

            
class ContextDependentValueBased2AFC(ValueBasedDecisionMaking):
    def __init__(
            self,
            choice_items: list[ChoiceItem], 
            choice_names: list[str],
            target_dtype: torch.dtype,
            zscore_domain: bool = False
        ):
        super().__init__(
            choice_items=choice_items, 
            choice_names=choice_names, 
            target_dtype=target_dtype,
            zscore_domain=zscore_domain
        )
        self.context_values = torch.tensor(
            [self.choices['values'][i_sample][0][0] for i_sample in range(self.__len__())]
        )

    def _valid_choice_condition(self, choice):
        context = choice[0][0]
        feature_0_diff = choice[1][0] - choice[2][0]
        feature_1_diff = choice[1][1] - choice[2][1]

        return False if (
            (context == 0 and feature_0_diff == 0)
            or (context == 1 and feature_1_diff == 0)
            or (feature_0_diff == 0 and feature_1_diff == 0)
        ) else True

    def _target_function(self, choice):
        if choice[0] == (0,):
            idx_relevant_feature = 0
            idx_irrelevant_feature = 1
        elif choice[0] == (1,):
            idx_relevant_feature = 1
            idx_irrelevant_feature = 0
        else:
            raise ValueError(f"Unexpected value of {choice[0]} for choice item 1.")
    
        return (
            choice[1][idx_relevant_feature] - choice[2][idx_relevant_feature],
            choice[1][idx_irrelevant_feature] - choice[2][idx_irrelevant_feature]
        )
    
    def _zscore_domain(self, domain):
        domain[:, 2:] = (domain[:, 2:] - torch.mean(domain[:, 2:])) / torch.std(domain[:, 2:])
        return domain
    
    def get_sample_ind(self, trial_type: str):
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
                idx for idx in range(self.__len__()) 
                if comparison(self.relevant_differences[idx] * self.irrelevant_differences[idx])
            ]
        elif trial_type[:-1] == 'context:':
            try:
                context_idx = int(trial_type[-1])
            except ValueError as e:
                raise ValueError(
                    f"Tried to call `int` on final element of `trial_type` and got a ValueError: {e}. "
                    "Ensure final element is string representation of an integer."
                )
            return [
                idx for idx in range(self.__len__())
                if self.choices['values'][idx][0] == (context_idx,)
            ]
        
    def compute_context_specific_cross_covariance_operators(self, print: bool = False):
        # Compute cross_cov_operators for each context.
        num_contexts = len(set(self.context_values))
        num_input_features = self.domain.shape[1]
        num_output_features = len(self.target.shape)
        context_dependent_cross_cov_operators = torch.zeros((num_contexts, num_output_features, num_input_features))
        for i_context, context in enumerate(['0', '1']):
            trial_ind = self.get_sample_ind(trial_type='context:' + context)
            context_dependent_cross_cov_operators[i_context, :, :] = utils.compute_cross_covariance_operator(
                self.domain[trial_ind], 
                self.target[trial_ind]
            )
        
        if print:
            print(f"tensor of context dependent cross cov operators: {context_dependent_cross_cov_operators}.")
            print(f"tensor of context dependent cross cov operators has shape: {context_dependent_cross_cov_operators.shape}.")

        return context_dependent_cross_cov_operators
    