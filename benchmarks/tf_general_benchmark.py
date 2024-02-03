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


def piquasso_benchmark(cutoff, alpha, r, xi):
    profiler_options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
    )

    tf.profiler.experimental.start("logdir", options=profiler_options)

    alpha_ = tf.Variable(alpha)

    with pq.Program() as program:
        pq.Q(all) | pq.Vacuum()

        pq.Q(0) | pq.Displacement(r=alpha_)
        pq.Q(1) | pq.Displacement(r=alpha_)
        pq.Q(0) | pq.Squeezing(r)
        pq.Q(1) | pq.Squeezing(r)
        pq.Q(all) | pq.Interferometer(unitary_group.rvs(2))
        pq.Q(0) | pq.Kerr(xi)
        pq.Q(1) | pq.Kerr(xi)

    simulator_fock = pq.PureFockSimulator(
        d=2,
        config=pq.Config(cutoff=cutoff),
        calculator=pq.TensorflowCalculator(),
    )

    with tf.GradientTape() as tape:
        state = simulator_fock.execute(program).state
        mean_position = state.mean_position(0)

    tape.gradient(mean_position, [alpha_])

    tf.profiler.experimental.stop()


def strawberryfields_benchmark(cutoff, alpha, r, xi):
    profiler_options = tf.profiler.experimental.ProfilerOptions(
        host_tracer_level=3, python_tracer_level=3, device_tracer_level=3
    )

    tf.profiler.experimental.start("logdir", options=profiler_options)

    program = sf.Program(2)

    mapping = {}

    alpha_ = tf.Variable(alpha)
    param = program.params("alpha")
    mapping["alpha"] = alpha_

    engine = sf.Engine(backend="tf", backend_options={"cutoff_dim": cutoff})

    with program.context as q:
        for i in range(2):
            sf.ops.Dgate(param) | q[i]
            sf.ops.Sgate(r) | q[i]

        sf.ops.Interferometer(unitary_group.rvs(2)) | tuple(q[i] for i in range(2))

        for i in range(2):
            sf.ops.Kgate(xi) | q[i]

    with tf.GradientTape() as tape:
        result = engine.run(program, args=mapping)
        state = result.state
        mean = sum([state.quad_expectation(mode)[0] for mode in range(2)])

    tape.gradient(mean, [alpha_])

    tf.profiler.experimental.stop()
