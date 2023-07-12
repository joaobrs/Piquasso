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

import pytest

import piquasso as pq
import strawberryfields as sf
import tensorflow as tf

from scipy.stats import unitary_group


pytestmark = pytest.mark.benchmark(
    group="tf-general",
)


@pytest.fixture
def alpha():
    return 0.01


@pytest.fixture
def r():
    return 0.01


@pytest.fixture
def xi():
    return 0.3


@pytest.fixture
def cutoff():
    return 10


@pytest.fixture
def layers():
    return 10

@pytest.fixture
def d():
    return 2


def piquasso_benchmark(benchmark, cutoff, alpha, r, xi, layers, d):
    @benchmark
    def func():
        alpha_ = tf.Variable(alpha)

        with pq.Program() as program:
            pq.Q(all) | pq.Vacuum()

            for _ in range(layers):
                for i in range(d):
                    pq.Q(i) | pq.Displacement(r=alpha_)
                    pq.Q(i) | pq.Squeezing(r)

                pq.Q(all) | pq.Interferometer(unitary_group.rvs(d))

                for i in range(d):
                    pq.Q(i) | pq.Kerr(xi)

        simulator_fock = pq.TensorflowPureFockSimulator(
            d=d, config=pq.Config(cutoff=cutoff, normalize=False)
        )

        with tf.GradientTape() as tape:
            state = simulator_fock.execute(program).state
            mean_position = [state.mean_position(i) for i in range(d)]

        tape.gradient(mean_position, [alpha_])


def strawberryfields_benchmark(benchmark, cutoff, alpha, r, xi, layers, d):
    @benchmark
    def func():
        program = sf.Program(d)

        mapping = {}

        alpha_ = tf.Variable(alpha)
        param = program.params("alpha")
        mapping["alpha"] = alpha_

        engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})

        with program.context as q:
            for _ in range(layers):
                for i in range(d):
                    sf.ops.Dgate(param) | q[i]
                    sf.ops.Sgate(r) | q[i]

                sf.ops.Interferometer(unitary_group.rvs(d)) | tuple(q[i] for i in range(d))

                for i in range(d):
                    sf.ops.Kgate(xi) | q[i]

        with tf.GradientTape() as tape:
            result = engine.run(program, args=mapping)
            state = result.state
            mean = sum([state.quad_expectation(mode)[0] for mode in range(d)])

        tape.gradient(mean, [alpha_])
