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

"""Implementation for the Clements decomposition.

References
~~~~~~~~~~

William R. Clements, Peter C. Humphreys, Benjamin J. Metcalf,
W. Steven Kolthammer, Ian A. Walmsley, "An Optimal Design for Universal Multiport
Interferometers", `arXiv:1603.08788 <https://arxiv.org/abs/1603.08788>`_.
"""

from typing import List, Tuple, TYPE_CHECKING

from dataclasses import dataclass

from piquasso.api.calculator import BaseCalculator

from piquasso._math.indices import get_operator_index

from piquasso._backends.calculator import NumpyCalculator

if TYPE_CHECKING:
    import numpy as np


@dataclass
class BS:
    """
    The Beamsplitter is implemented as described in
    `arXiv:1603.08788 <https://arxiv.org/abs/1603.08788>`_.
    """

    modes: Tuple[int, int]
    params: Tuple["np.float64", "np.float64"]


@dataclass
class PS:
    """
    Phaseshifter gate.
    """

    mode: int
    phi: "np.float64"


@dataclass
class Decomposition:
    """
    The data stucture which holds the decomposed angles from the Clements decomposition.

    Example usage::

        with pq.Program() as program_with_decomposition:
            ...

            for operation in decomposition.first_beamsplitters:
                pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=operation.params[1])
                pq.Q(*operation.modes) | pq.Beamsplitter(operation.params[0], 0.0)

            for operation in decomposition.middle_phaseshifters:
                pq.Q(operation.mode) | pq.Phaseshifter(operation.phi)

            for operation in decomposition.last_beamsplitters:
                pq.Q(*operation.modes) | pq.Beamsplitter(-operation.params[0], 0.0)
                pq.Q(operation.modes[0]) | pq.Phaseshifter(phi=-operation.params[1])

    """

    first_beamsplitters: List[BS]
    middle_phaseshifters: List[PS]
    last_beamsplitters: List[BS]


def inverse_clements(
    decomposition: Decomposition, calculator: BaseCalculator, dtype: "np.dtype"
) -> "np.ndarray":
    """Inverse of the Clements decomposition.

    Returns:
        The unitary matrix corresponding to the interferometer.
    """

    d = len(decomposition.middle_phaseshifters)

    np = calculator.np
    interferometer = np.identity(d, dtype=dtype)

    for beamsplitter in decomposition.first_beamsplitters:
        beamsplitter_matrix = _get_embedded_beamsplitter_matrix(
            beamsplitter, d, calculator, dtype=dtype
        )

        interferometer = beamsplitter_matrix @ interferometer

    phis = np.empty(d, dtype=interferometer.dtype)

    for phaseshifter in decomposition.middle_phaseshifters:
        phis = calculator.assign(phis, phaseshifter.mode, phaseshifter.phi)

    interferometer = np.diag(np.exp(1j * phis)) @ interferometer

    for beamsplitter in decomposition.last_beamsplitters:
        beamsplitter_matrix = np.conj(
            _get_embedded_beamsplitter_matrix(beamsplitter, d, calculator, dtype=dtype)
        ).T

        interferometer = beamsplitter_matrix @ interferometer

    return interferometer


def clements(
    U: "np.ndarray",
    calculator: BaseCalculator,
) -> Decomposition:
    """
    Decomposes the specified unitary matrix by application of beamsplitters
    prescribed by the decomposition.

    Args:
        U (numpy.ndarray): The (square) unitary matrix to be decomposed.

    Returns:
        The Clements decomposition. See :class:`Decomposition`.
    """

    first_beampsplitters = []
    last_beamsplitters = []

    np = calculator.np

    d = U.shape[0]

    for column in reversed(range(0, d - 1)):
        if column % 2 == 0:
            operations, U = _apply_direct_beamsplitters(column, U, calculator)
            last_beamsplitters.extend(operations)
        else:
            operations, U = _apply_inverse_beamsplitters(column, U, calculator)
            first_beampsplitters.extend(operations)

    middle_phasshifters = [
        PS(mode=mode, phi=np.angle(diagonal))
        for mode, diagonal in enumerate(np.diag(U))
    ]

    last_beamsplitters = list(reversed(last_beamsplitters))

    return Decomposition(
        first_beamsplitters=first_beampsplitters,
        last_beamsplitters=last_beamsplitters,
        middle_phaseshifters=middle_phasshifters,
    )


def _apply_direct_beamsplitters(
    column: int, U: "np.ndarray", calculator: BaseCalculator
) -> tuple:
    """
    Calculates the direct beamsplitters for a given column `column`, and
    applies it to `U`.

    Args:
        column (int): The current column.
    """

    operations = []

    d = U.shape[0]

    dtype = U.dtype

    for j in range(d - 1 - column):
        modes = (column + j, column + j + 1)

        matrix_element_to_eliminate = U[modes[0], j]
        matrix_element_above = -U[modes[1], j]

        angles = _get_angles(
            matrix_element_to_eliminate, matrix_element_above, calculator
        )

        operation = BS(modes=modes, params=angles)

        matrix = _get_embedded_beamsplitter_matrix(operation, d, calculator, dtype)

        U = matrix @ U

        operations.append(operation)

    return operations, U


def _apply_inverse_beamsplitters(
    column: int, U: "np.ndarray", calculator: BaseCalculator
) -> tuple:
    """
    Calculates the inverse beamsplitters for a given column `column`, and
    applies it to `U`.

    Args:
        column (int): The current column.
    """

    operations = []

    d = U.shape[0]

    dtype = U.dtype

    for j in reversed(range(d - 1 - column)):
        modes = (j, j + 1)

        i = column + j + 1

        matrix_element_to_eliminate = U[i, modes[1]]
        matrix_element_to_left = U[i, modes[0]]

        angles = _get_angles(
            matrix_element_to_eliminate, matrix_element_to_left, calculator
        )

        operation = BS(modes=modes, params=angles)

        beamsplitter = calculator.np.conj(
            _get_embedded_beamsplitter_matrix(operation, d, calculator, dtype)
        ).T

        U = U @ beamsplitter

        operations.append(operation)

    return operations, U


def _get_angles(matrix_element_to_eliminate, other_matrix_element, calculator):
    np = calculator.np

    if np.isclose(matrix_element_to_eliminate, 0.0):
        return np.pi / 2, 0.0

    r = other_matrix_element / matrix_element_to_eliminate
    theta = np.arctan(np.abs(r))
    phi = np.angle(r)

    return theta, phi


def _get_embedded_beamsplitter_matrix(
    operation: BS, d: int, calculator: BaseCalculator, dtype: "np.dtype"
) -> "np.ndarray":
    np = calculator.np

    theta, phi = operation.params
    i, j = operation.modes

    c = np.cos(theta).astype(dtype)
    s = np.sin(theta).astype(dtype)

    matrix = np.array(
        [
            [np.exp(1j * phi) * c, -s],
            [np.exp(1j * phi) * s, c],
        ],
        dtype=dtype,
    )

    return calculator.embed_in_identity(matrix, get_operator_index((i, j)), d)


def get_weights_from_decomposition(
    decomposition: Decomposition, d: int, calculator: BaseCalculator
) -> "np.ndarray":
    """Concatenates the weight vector from the angles in the Clements decomposition.

    Returns:
        The Clements decomposition. See :class:`Decomposition`.
    """
    np = calculator.np

    dtype = decomposition.middle_phaseshifters[0].phi.dtype
    weights = np.empty(d**2, dtype=dtype)

    index = 0
    for beamsplitter in decomposition.first_beamsplitters:
        weights = calculator.assign(weights, index, beamsplitter.params[0])
        index += 1
        weights = calculator.assign(weights, index, beamsplitter.params[1])
        index += 1

    for phaseshifter in decomposition.middle_phaseshifters:
        weights = calculator.assign(weights, index, phaseshifter.phi)
        index += 1

    for beamsplitter in decomposition.last_beamsplitters:
        weights = calculator.assign(weights, index, beamsplitter.params[0])
        index += 1
        weights = calculator.assign(weights, index, beamsplitter.params[1])
        index += 1

    return weights


def get_decomposition_from_weights(
    weights: "np.ndarray", d: int, calculator: BaseCalculator
) -> Decomposition:
    """Puts the data in the weight vector into a Clements decompositon.

    Returns:
        The Clements decomposition. See :class:`Decomposition`.
    """

    fallback_np = calculator.fallback_np

    # NOTE: This is tricky: the ordering in the Clements decomposition is not unique,
    # since beamsplitters acting on different modes may commute, and the ordering comes
    # out very ugly after all the Givens rotations. Therefore, it is easier to just
    # create a trivial decomposition, and fill it with the required values (for now).
    decomposition = clements(fallback_np.identity(d), calculator=NumpyCalculator())

    index = 0

    for beamsplitter in decomposition.first_beamsplitters:
        beamsplitter.params = (weights[index], weights[index + 1])
        index += 2

    for phaseshifter in decomposition.middle_phaseshifters:
        phaseshifter.phi = weights[index]
        index += 1

    for beamsplitter in decomposition.last_beamsplitters:
        beamsplitter.params = (weights[index], weights[index + 1])
        index += 2

    return decomposition


def get_weights_from_interferometer(
    U: "np.ndarray", calculator: BaseCalculator
) -> "np.ndarray":
    """Creates a vector of weights from the Clements angles."""
    decomposition = clements(U, calculator)

    return get_weights_from_decomposition(decomposition, U.shape[0], calculator)


def get_interferometer_from_weights(
    weights: "np.ndarray",
    d: int,
    calculator: BaseCalculator,
    dtype: "np.dtype",
) -> "np.ndarray":
    """Returns the interferometer matrix corresponding to the specified weights.

    It is the inverse of :func:`get_weights_from_interferometer`.
    """
    decomposition = get_decomposition_from_weights(weights, d, calculator)

    return inverse_clements(decomposition, calculator, dtype)
