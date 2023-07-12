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

import pytest

import piquasso as pq
import strawberryfields as sf

from scipy.stats import unitary_group


pytestmark = pytest.mark.benchmark(
    group="pure-fock-general",
)


@pytest.fixture
def alpha():
    return 0.1


@pytest.fixture
def r():
    return 0.2


@pytest.fixture
def xi():
    return 0.3


@pytest.fixture
def cutoff():
    return 40


def piquasso_benchmark(cutoff, r, alpha, xi):
    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        for i in range(2):
            pq.Q(i) | pq.Displacement(r=alpha) | pq.Squeezing(r)

        pq.Q(all) | pq.Interferometer(unitary_group.rvs(2))

        for i in range(2):
            pq.Q(i) | pq.Kerr(xi)

    simulator_fock = pq.PureFockSimulator(d=2, config=pq.Config(cutoff=cutoff))

    simulator_fock.execute(program)


def strawberryfields_benchmark(cutoff, r, alpha, xi):
    eng = sf.Engine(backend="fock", backend_options={"cutoff_dim": cutoff})

    circuit = sf.Program(2)

    with circuit.context as q:
        for i in range(2):
            sf.ops.Dgate(alpha) | q[i]
            sf.ops.Sgate(r) | q[i]

        sf.ops.Interferometer(unitary_group.rvs(2)) | tuple(q[i] for i in range(2))

        for i in range(2):
            sf.ops.Kgate(xi) | q[i]

    eng.run(circuit)
