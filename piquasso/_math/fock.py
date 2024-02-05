#
# Copyright 2021-2024 Budapest Quantum Computing Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from typing import Optional, Tuple, Iterable, Generator, Any, List

import numpy as np
from operator import add

from scipy.special import factorial, comb

from piquasso.api.config import Config
from piquasso._math.indices import get_operator_index
from piquasso._math.combinatorics import partitions
from piquasso._math.gradients import (
    create_single_mode_displacement_gradient,
    create_single_mode_squeezing_gradient,
)
from piquasso._math.gate_matrices import (
    create_single_mode_displacement_matrix,
    create_single_mode_squeezing_matrix,
)
from piquasso.api.calculator import BaseCalculator


@functools.lru_cache()
def cutoff_cardinality(*, cutoff: int, d: int) -> int:
    r"""
    Calculates the dimension of the cutoff Fock space with the relation

    ..math::
        \sum_{i=0}^{c - 1} {d + i - 1 \choose i} = {d + c - 1 \choose c - 1}.
    """
    return comb(d + cutoff - 1, cutoff - 1, exact=True)


@functools.lru_cache()
def symmetric_subspace_cardinality(*, d: int, n: int) -> int:
    return comb(d + n - 1, n, exact=True)


@functools.lru_cache(maxsize=None)
def _create_all_fock_basis_elements(d: int, cutoff: int) -> List[Tuple[int, ...]]:
    ret = []

    for n in range(cutoff):
        ret.extend(partitions(boxes=d, particles=n, class_=FockBasis))

    return ret


class FockBasis(tuple):
    def __str__(self) -> str:
        return self.display(template="|{}>")

    def display(self, template: str = "{}") -> str:
        return template.format("".join([str(number) for number in self]))

    def display_as_bra(self) -> str:
        return self.display("<{}|")

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, other: Iterable) -> "FockBasis":
        return FockBasis(map(add, self, other))

    __radd__ = __add__

    @property
    def d(self) -> int:
        return len(self)

    @property
    def n(self) -> int:
        return sum(self)

    @classmethod
    def create_all(cls, *, d: int, cutoff: int) -> List[Tuple[int, ...]]:
        return _create_all_fock_basis_elements(d, cutoff)

    def on_modes(self, *, modes: Tuple[int, ...]) -> "FockBasis":
        return FockBasis(self[mode] for mode in modes)

    def increment_on_modes(self, modes: Tuple[int, ...]) -> "FockBasis":
        a = [0] * self.d
        for mode in modes:
            a[mode] = 1

        return self + a


class FockOperatorBasis(tuple):
    def __new__(cls, *, ket: Iterable, bra: Iterable) -> "FockOperatorBasis":
        return super().__new__(cls, (FockBasis(ket), FockBasis(bra)))  # type: ignore

    def __str__(self) -> str:
        return str(self.ket) + self.bra.display_as_bra()

    @property
    def ket(self) -> FockBasis:
        return self[0]

    @property
    def bra(self) -> FockBasis:
        return self[1]

    def is_diagonal_on_modes(self, modes: Tuple[int, ...]) -> bool:
        return self.ket.on_modes(modes=modes) == self.bra.on_modes(modes=modes)


class FockSpace(tuple):
    r"""
    Note, that when you symmetrize a tensor, i.e. use the superoperator

    .. math::
        S: A \otimes B \mapsto A \vee B

    on a tensor which is expressed in the regular Fock basis, then the resulting tensor
    still remains in the regular representation. You have to perform a basis
    transformation to acquire the symmetrized tensor in the symmetrized representation.
    """

    def __new__(
        cls, d: int, cutoff: int, calculator: BaseCalculator, config: Config
    ) -> "FockSpace":
        return super().__new__(
            cls, FockBasis.create_all(d=d, cutoff=cutoff)  # type: ignore
        )

    def __init__(
        self, *, d: int, cutoff: int, calculator: BaseCalculator, config: Config
    ) -> None:
        self.d = d
        self.cutoff = cutoff
        self.calculator = calculator
        self.config = config
        self._calculator = calculator

    def __deepcopy__(self, memo: Any) -> "FockSpace":
        """
        This method exists, because `copy.deepcopy` produces errors with classes
        defining both `__new__` and `__init__`.

        It defines the deepcopy of this object. Since its state (:attr:`d` and
        :attr:`cutoff`) is immutable, we don't really need to deepcopy this object, we
        could return with this instance, too.
        """

        return self

    def get_passive_fock_operator(
        self,
        operator: np.ndarray,
        modes: Tuple[int, ...],
        d: int,
    ) -> np.ndarray:
        indices = get_operator_index(modes)

        embedded_operator = self._calculator.embed_in_identity(operator, indices, d)

        return self._calculator.block_diag(
            *(
                self.symmetric_tensorpower(embedded_operator, n)
                for n in range(self.cutoff)
            )
        )

    def get_single_mode_squeezing_operator(
        self,
        *,
        r: float,
        phi: float,
    ) -> np.ndarray:
        @self.calculator.custom_gradient
        def _single_mode_squeezing_operator(r, phi):
            r = self.calculator.maybe_convert_to_numpy(r)
            phi = self.calculator.maybe_convert_to_numpy(phi)

            matrix = create_single_mode_squeezing_matrix(
                r, phi, self.cutoff, complex_dtype=self.config.complex_dtype
            )
            grad = create_single_mode_squeezing_gradient(
                r,
                phi,
                self.cutoff,
                matrix,
                self.calculator,
            )
            return matrix, grad

        return _single_mode_squeezing_operator(r, phi)

    def get_single_mode_cubic_phase_operator(
        self, *, gamma: float, hbar: float, calculator: BaseCalculator
    ) -> np.ndarray:
        r"""Cubic Phase gate.

        The definition of the Cubic Phase gate is

        .. math::
            \operatorname{CP}(\gamma) = e^{i \hat{x}^3 \frac{\gamma}{3 \hbar}}

        The Cubic Phase gate transforms the annihilation operator as

        .. math::
            \operatorname{CP}^\dagger(\gamma) \hat{a} \operatorname{CP}(\gamma) =
                \hat{a} + i\frac{\gamma(\hat{a} +\hat{a}^\dagger)^2}{2\sqrt{2/\hbar}}

        It transforms the :math:`\hat{p}` quadrature as follows:

        .. math::
            \operatorname{CP}^\dagger(\gamma) \hat{p} \operatorname{CP}(\gamma) =
                \hat{p} + \gamma \hat{x}^2.

        Args:
            gamma (float): The Cubic Phase parameter.
            hbar (float): Scaling parameter.
        Returns:
            np.ndarray:
                The resulting transformation, which could be applied to the state.
        """

        np = calculator.np

        annih = np.diag(np.sqrt(np.arange(1, self.cutoff)), 1)
        position = (annih.T + annih) * np.sqrt(hbar / 2)
        return calculator.expm(1j * calculator.powm(position, 3) * (gamma / (3 * hbar)))

    @property
    def cardinality(self) -> int:
        return cutoff_cardinality(cutoff=self.cutoff, d=self.d)

    @property
    def basis(self) -> Generator[Tuple[int, FockBasis], Any, None]:
        yield from enumerate(self)

    @property
    def operator_basis(
        self,
    ) -> Generator[Tuple[Tuple[int, int], FockOperatorBasis], Any, None]:
        for index, basis in self.basis:
            for dual_index, dual_basis in self.basis:
                yield (index, dual_index), FockOperatorBasis(ket=basis, bra=dual_basis)

    def operator_basis_diagonal_on_modes(
        self, *, modes: Tuple[int, ...]
    ) -> Generator[Tuple[Tuple[int, int], FockOperatorBasis], Any, None]:
        yield from [
            (index, basis)
            for index, basis in self.operator_basis
            if basis.is_diagonal_on_modes(modes=modes)
        ]

    def get_occupied_basis(
        self, *, modes: Tuple[int, ...], occupation_numbers: Tuple[int, ...]
    ) -> FockBasis:
        temp = [0] * self.d
        for index, mode in enumerate(modes):
            temp[mode] = occupation_numbers[index]

        return FockBasis(temp)

    def get_projection_operator_indices_for_pure(
        self, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
    ) -> List[int]:
        return [
            index
            for index, basis in self.basis
            if subspace_basis == basis.on_modes(modes=modes)
        ]

    def get_projection_operator_indices(
        self, *, subspace_basis: Tuple[int, ...], modes: Tuple[int, ...]
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return tuple(  # type: ignore
            zip(
                *[
                    index
                    for index, operator_basis in self.operator_basis
                    if operator_basis.is_diagonal_on_modes(modes=modes)
                    and subspace_basis == operator_basis.ket.on_modes(modes=modes)
                ]
            )
        )

    def get_subspace_basis(
        self, n: int, d: Optional[int] = None
    ) -> List[Tuple[int, ...]]:
        d = d or self.d
        return partitions(boxes=d, particles=n, class_=FockBasis)

    def symmetric_tensorpower(
        self,
        operator: np.ndarray,
        n: int,
    ) -> np.ndarray:
        d = len(operator)

        np = self.calculator.np

        dim = symmetric_subspace_cardinality(d=d, n=n)

        matrix: List[List[complex]] = [[] for _ in range(dim)]

        subspace_operator_basis = self.get_subspace_basis(n, d)

        for row, row_basis in enumerate(subspace_operator_basis):
            for col_basis in subspace_operator_basis:
                cell = self.calculator.permanent(
                    operator,
                    col_basis,
                    row_basis,
                ) / np.sqrt(
                    np.prod(factorial(col_basis)) * np.prod(factorial(row_basis))
                )

                matrix[row].append(cell)

        return np.array(matrix)

    def get_creation_operator(self, modes: Tuple[int, ...]) -> np.ndarray:
        operator = np.zeros(
            shape=(self.cardinality,) * 2, dtype=self.config.complex_dtype
        )

        for index, basis in enumerate(self):
            dual_basis = basis.increment_on_modes(modes)
            try:
                dual_index = self.index(dual_basis)
                operator[dual_index, index] = 1
            except ValueError:
                # TODO: rethink.
                continue

        return operator

    def get_annihilation_operator(self, modes: Tuple[int, ...]) -> np.ndarray:
        return self.get_creation_operator(modes).transpose()

    def get_single_mode_displacement_operator(self, *, r, phi):
        @self.calculator.custom_gradient
        def _single_mode_displacement_operator(r, phi):
            r = self.calculator.maybe_convert_to_numpy(r)
            phi = self.calculator.maybe_convert_to_numpy(phi)

            matrix = create_single_mode_displacement_matrix(
                r, phi, self.cutoff, complex_dtype=self.config.complex_dtype
            )
            grad = create_single_mode_displacement_gradient(
                r,
                phi,
                self.cutoff,
                matrix,
                self.calculator,
            )
            return matrix, grad

        return _single_mode_displacement_operator(r, phi)
