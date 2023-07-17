#
# Copyright 2021-2023 Budapest Quantum Computing Group
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


def test_batch_Beamsplitter_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

    with pq.Program() as batch_program:
        pq.Q() | pq.Batch([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state._state_vector

    first_state_vector = simulator.execute(first_program).state._state_vector
    second_state_vector = simulator.execute(second_program).state._state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_Squeezing_and_Displacement_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Squeezing(0.1)

    with pq.Program() as batch_program:
        pq.Q() | pq.Batch([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state._state_vector

    first_state_vector = simulator.execute(first_program).state._state_vector
    second_state_vector = simulator.execute(second_program).state._state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_Kerr_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0) | pq.Kerr(0.1)
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.Batch([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state._state_vector

    first_state_vector = simulator.execute(first_program).state._state_vector
    second_state_vector = simulator.execute(second_program).state._state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_Phaseshifter_state_vector():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0) | pq.Phaseshifter(0.1)
        pq.Q(1) | pq.Phaseshifter(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.Batch([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_state_vector = simulator.execute(batch_program).state._state_vector

    first_state_vector = simulator.execute(first_program).state._state_vector
    second_state_vector = simulator.execute(second_program).state._state_vector

    assert np.allclose(batch_state_vector[:, 0], first_state_vector)
    assert np.allclose(batch_state_vector[:, 1], second_state_vector)


def test_batch_mean_position():
    with pq.Program() as first_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(1) | pq.Squeezing(r=0.1, phi=np.pi / 4)

    with pq.Program() as second_preparation:
        pq.Q() | pq.Vacuum()
        pq.Q(0) | pq.Displacement(r=1.0, phi=np.pi / 10)

    with pq.Program() as common:
        pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 6, phi=np.pi / 3)

        pq.Q(0) | pq.Displacement(1.0)
        pq.Q(0) | pq.Squeezing(0.1)

        pq.Q(0) | pq.Kerr(0.1)
        pq.Q(1) | pq.Kerr(0.2)

    with pq.Program() as batch_program:
        pq.Q() | pq.Batch([first_preparation, second_preparation])

        pq.Q() | common

    with pq.Program() as first_program:
        pq.Q() | first_preparation

        pq.Q() | common

    with pq.Program() as second_program:
        pq.Q() | second_preparation

        pq.Q() | common

    simulator = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=5))

    batch_mean_positions = simulator.execute(batch_program).state.mean_position(0)

    first_mean_position = simulator.execute(first_program).state.mean_position(0)
    second_mean_position = simulator.execute(second_program).state.mean_position(0)

    assert np.allclose(batch_mean_positions[0], first_mean_position)
    assert np.allclose(batch_mean_positions[1], second_mean_position)
