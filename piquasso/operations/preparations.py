#
# Copyright (C) 2020 by TODO - All rights reserved.
#


from piquasso.core.mixins import _WeightMixin
from piquasso.core.registry import _register
from piquasso.api.operation import Operation


@_register
class Vacuum(Operation):
    r"""Prepare the system in a vacuum state."""

    def __init__(self):
        pass


@_register
class Mean(Operation):
    r"""Set the first canonical moment of the state."""

    def __init__(self, mean):
        super().__init__(mean)


@_register
class Covariance(Operation):
    r"""Sets the covariance matrix of the state."""

    def __init__(self, cov):
        super().__init__(cov)


@_register
class Number(Operation, _WeightMixin):
    r"""State preparation with Fock basis vectors."""

    def __init__(self, *occupation_numbers, ket=None, bra=None, coefficient=1.0):

        if occupation_numbers:
            ket = bra = occupation_numbers

        super().__init__(ket, bra, coefficient)


@_register
class Create(Operation):
    r"""Create a particle on a mode."""

    def __init__(self):
        pass


@_register
class Annihilate(Operation):
    r"""Annihilate a particle on a mode."""

    def __init__(self):
        pass
