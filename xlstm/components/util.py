# Copyright (c) NXAI GmbH and its affiliates 2024
# Korbininan Pöppel
import torch
from typing import Callable


def round_to_multiple(n, m=8):
    return ((n + m - 1) // m) * m


def conditional_decorator(condition, decorator):
    """A higher-order decorator that applies 'decorator' only if 'condition' is True."""

    def dummy_decorator(func):
        """A dummy decorator that does nothing."""
        return func

    if condition:
        # If condition is True, return the actual decorator
        return decorator
    else:
        # If condition is False, return the dummy decorator
        return dummy_decorator


class ParameterProxy:
    """
    This class helps keeping parameters in a specialized internal structure to be optimal for
    computation speed, while having a proxied version to be called externally that is backend-agnostic.
    It takes a module and a parameter name of a parameter in that module it represents.
    Via __setitem__ and __getitem__ the "external"
    """

    def __init__(
        self,
        module,
        parameter_name,
        internal_to_external: Callable[[torch.Tensor], torch.Tensor],
        external_to_internal: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.module = module
        self.parameter_name = parameter_name
        self.internal_to_external = internal_to_external
        self.external_to_internal = external_to_internal

    def __getitem__(self, key):
        # Transform and then apply the slice to the external shape
        external_param = self.internal_to_external(getattr(self.module, self.parameter_name)).detach()
        return external_param[key]

    def __setitem__(self, key, value):
        # Apply the slice on the external shape, then transform back
        with torch.no_grad():
            external_param = self.internal_to_external(getattr(self.module, self.parameter_name))
            external_param[key] = value
            getattr(self.module, self.parameter_name).data = self.external_to_internal(external_param).contiguous()

    def clone(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).clone()

    @property
    def shape(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).shape

    @property
    def ndim(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name)).ndim

    @property
    def grad(self):
        return self.internal_to_external(getattr(self.module, self.parameter_name).grad)

    def __getattr__(self, name: str):
        return getattr(getattr(self.module, self.parameter_name), name)
