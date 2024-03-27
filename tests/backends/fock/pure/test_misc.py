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

import numpy as np
import piquasso as pq

from scipy.stats import unitary_group


def test_post_select_NS_gate():
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

        pq.Q(all) | pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    assert np.isclose(state.norm, 0.25)

    reduced_state = state.reduced(modes=(0,))

    assert np.isclose(reduced_state.norm, 0.25)

    reduced_state.normalize()
    reduced_state.validate()

    purity = reduced_state.get_purity()

    assert np.isclose(purity, 1.0)

    expected_state_vector = np.copy(first_mode_state_vector)
    expected_state_vector[2] *= -1

    assert np.allclose(
        reduced_state.density_matrix[:d, :d],
        np.outer(expected_state_vector, expected_state_vector),
    )


def test_post_select_random_unitary():
    d = 3

    interferometer_matrix = unitary_group.rvs(d)

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * np.sqrt(0.2)
        pq.Q(all) | pq.StateVector([1, 1, 0]) * np.sqrt(0.3)
        pq.Q(all) | pq.StateVector([2, 1, 0]) * np.sqrt(0.5)

        pq.Q(all) | pq.Interferometer(interferometer_matrix)

        pq.Q(all) | pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    state.normalize()
    state.validate()


def test_post_select_conditional_sign_flip_gate_with_1_over_16_success_rate():
    modes = (0, 1, 2, 3)
    ancilla_modes = (4, 5, 6, 7)

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    with pq.Program() as nonlinear_shift:
        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

    with pq.Program() as conditional_sign_flip:
        pq.Q(0, 2) | pq.Beamsplitter(theta=np.pi / 4)

        pq.Q(0, ancilla_modes[0], ancilla_modes[1]) | nonlinear_shift
        pq.Q(2, ancilla_modes[2], ancilla_modes[3]) | nonlinear_shift

        pq.Q(0, 2) | pq.Beamsplitter(theta=-np.pi / 4)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    states = [state_00, state_01, state_10, state_11]
    coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])

    ancilla_state = [1, 0]

    with pq.Program() as program:
        for input_state, coeff in zip(states, coefficients):
            pq.Q(all) | pq.StateVector(input_state + ancilla_state * 2) * coeff

        pq.Q(all) | conditional_sign_flip

        pq.Q(all) | pq.PostSelectPhotons(
            postselect_modes=ancilla_modes, photon_counts=ancilla_state * 2
        )

    simulator = pq.PureFockSimulator(d=8, config=pq.Config(cutoff=5))

    final_state = simulator.execute(program).state.reduced(modes=modes)

    expected_success_rate = 1 / 16
    actual_success_rate = final_state.norm

    assert np.isclose(expected_success_rate, actual_success_rate)

    final_state.normalize()

    purity = final_state.get_purity()

    assert np.isclose(purity, 1.0)

    expected_coefficients = np.sqrt([0.1, 0.2, 0.3, 0.4])

    expected_coefficients[3] *= -1

    with pq.Program() as expectation_program:
        for input_state, coeff in zip(states, expected_coefficients):
            pq.Q(all) | pq.StateVector(input_state) * coeff

    expectation_simulator = pq.PureFockSimulator(d=4, config=pq.Config(cutoff=5))
    expected_state = expectation_simulator.execute(expectation_program).state

    assert np.allclose(expected_state.density_matrix, final_state.density_matrix)


def test_ImperfectPostSelectPhotons():
    d = 5
    cutoff = 4

    detector_efficiency_matrix = np.array(
        [
            [1.0, 0.2, 0.1],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 0.7],
        ]
    )

    coeffs = np.sqrt([0.1, 0.3, 0.4, 0.05, 0.1, 0.05])

    with pq.Program() as program:
        pq.Q() | pq.StateVector([0, 0, 0, 2, 1]) * coeffs[0]
        pq.Q() | pq.StateVector([0, 0, 2, 0, 1]) * coeffs[1]
        pq.Q() | pq.StateVector([0, 1, 0, 1, 1]) * coeffs[2]
        pq.Q() | pq.StateVector([1, 1, 0, 1, 0]) * coeffs[3]
        pq.Q() | pq.StateVector([3, 0, 0, 0, 0]) * coeffs[4]

        pq.Q() | pq.ImperfectPostSelectPhotons(
            postselect_modes=(2, 4),
            photon_counts=(0, 1),
            detector_efficiency_matrix=detector_efficiency_matrix,
        )

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=cutoff))

    reduced_state = simulator.execute(program).state

    with pq.Program() as expected_program:
        pq.Q() | pq.DensityMatrix((0, 1, 1), (0, 1, 1)) * 0.32
        pq.Q() | pq.DensityMatrix((0, 1, 1), (0, 0, 2)) * 0.16
        pq.Q() | pq.DensityMatrix((0, 0, 2), (0, 1, 1)) * 0.16
        pq.Q() | pq.DensityMatrix((0, 0, 2), (0, 0, 2)) * 0.08

    expected_simulator = pq.FockSimulator(d=3, config=pq.Config(cutoff=4))

    expected_reduced_state = expected_simulator.execute(expected_program).state

    assert reduced_state == expected_reduced_state


def test_NS_gate_with_ImperfectPostSelectPhotons_trivial_case():
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    trivial_detector_efficiency_matrix = np.identity(d)

    with pq.Program() as preparation:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

    with pq.Program() as imperfect_photon_detection_program:
        pq.Q(all) | preparation

        pq.Q(all) | pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=trivial_detector_efficiency_matrix,
        )

    with pq.Program() as perfect_photon_detection_program:
        pq.Q(all) | preparation

        pq.Q(all) | pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=4))

    imperfect_state = simulator.execute(imperfect_photon_detection_program).state
    perfect_state = simulator.execute(perfect_photon_detection_program).state

    imperfect_success_rate = imperfect_state.norm
    perfect_success_rate = perfect_state.norm
    assert np.isclose(imperfect_success_rate, perfect_success_rate)
    assert np.isclose(perfect_success_rate, 1 / 4)

    imperfect_state.normalize()
    perfect_state.normalize()

    imperfect_state.validate()
    perfect_state.validate()

    assert np.isclose(imperfect_state.get_purity(), 1.0)

    assert np.allclose(imperfect_state.density_matrix, perfect_state.density_matrix)


def test_NS_gate_with_ImperfectPostSelectPhotons():
    d = 3

    first_mode_state_vector = np.sqrt([0.2, 0.3, 0.5])

    ns_gate_interferometer = np.array(
        [
            [1 - 2 ** (1 / 2), 2 ** (-1 / 4), np.sqrt(3 / 2 ** (1 / 2) - 2)],
            [2 ** (-1 / 4), 1 / 2, 1 / 2 - np.sqrt(1 / 2)],
            [np.sqrt(3 / 2 ** (1 / 2) - 2), 1 / 2 - np.sqrt(1 / 2), np.sqrt(2) - 1 / 2],
        ]
    )

    detector_efficiency_matrix = np.array(
        [[1.0, 0.1, 0.2], [0.0, 0.9, 0.2], [0.0, 0.0, 0.6]]
    )

    with pq.Program() as program:
        pq.Q(all) | pq.StateVector([0, 1, 0]) * first_mode_state_vector[0]
        pq.Q(all) | pq.StateVector([1, 1, 0]) * first_mode_state_vector[1]
        pq.Q(all) | pq.StateVector([2, 1, 0]) * first_mode_state_vector[2]

        pq.Q(all) | pq.Interferometer(ns_gate_interferometer)

        pq.Q(all) | pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=detector_efficiency_matrix,
        )

    simulator = pq.PureFockSimulator(d=d, config=pq.Config(cutoff=4))

    state = simulator.execute(program).state

    success_rate = state.norm

    assert np.isclose(success_rate, 0.25773863607376246)

    state.normalize()
    state.validate()

    assert np.isclose(state.get_purity(), 0.8645456946151449)

    assert np.allclose(
        state.density_matrix,
        [
            [0.25690057, 0.25784735, -0.27605969, 0.0],
            [0.25784735, 0.30661073, -0.33810269, 0.0],
            [-0.27605969, -0.33810269, 0.43648869, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
    )
