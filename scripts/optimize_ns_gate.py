import piquasso as pq

import tensorflow as tf

from piquasso._math.fock import cutoff_cardinality


tf.get_logger().setLevel("ERROR")


def _calculate_loss(weights, calculator, state_vector):
    d = 3
    config = pq.Config(cutoff=4, normalize=False)
    np = calculator.np
    phase_shifter_phis = weights[:3]
    thetas = weights[3:6]
    phis = weights[6:]

    program = pq.Program(instructions=[
        pq.StateVector([0, 1, 0], coefficient=state_vector[0]),
        pq.StateVector([1, 1, 0], coefficient=state_vector[1]),
        pq.StateVector([2, 1, 0], coefficient=state_vector[2]),

        pq.Phaseshifter(phase_shifter_phis[0]).on_modes(0),
        pq.Phaseshifter(phase_shifter_phis[1]).on_modes(1),
        pq.Phaseshifter(phase_shifter_phis[2]).on_modes(2),

        pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(1, 2),
        pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(0, 1),
        pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(1, 2),

        pq.PostSelectPhotons(postselect_modes=(1, 2), photon_counts=(1, 0))
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    reduced_state = simulator.execute(program).state.reduced((0, ))

    density_matrix = reduced_state.density_matrix[:3, :3]

    norm = tf.math.reduce_sum(fock_probabilities(d=d, config=config, density_matrix=density_matrix))
    normalized_density_matrix = density_matrix / norm

    expected_state = state_vector
    expected_state = calculator.assign(expected_state, 2, -expected_state[2])

    target_density_matrix = np.outer(np.conj(expected_state), expected_state)

    # loss = tf.norm(density_matrix - target_density_matrix, ord=2)
    loss = 1 - tf.math.abs(np.conj(expected_state) @ normalized_density_matrix @ expected_state)

    return loss


def fock_probabilities(d, config, density_matrix):
    cardinality = cutoff_cardinality(d=d, cutoff=config.cutoff)

    return tf.math.real(tf.linalg.tensor_diag_part(density_matrix))[:cardinality]


def train_step(weights, calculator, state_vector):
    with tf.GradientTape() as tape:
        loss = _calculate_loss(
            weights=weights,
            calculator=calculator,
            state_vector=state_vector
        )

    grad = tape.gradient(loss, weights)

    return loss, grad


def main():
    opt = tf.keras.optimizers.Adam(learning_rate=0.0000010)
    decorator = tf.function(jit_compile=True)
    calculator = pq.TensorflowCalculator(decorate_with=decorator)
    np = calculator.np
    fallback_np = calculator.fallback_np
    fallback_np.random.seed(123)

    ideal_weights = fallback_np.array([np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, - np.pi / 8, 0, 0, 0])
    errors = fallback_np.random.normal(0, 0.1, size=9)
    weights = tf.Variable(ideal_weights + errors, dtype=tf.float64)

    state_vector = np.sqrt([0.2, 0.3, 0.5])

    with open("losses.csv", "a+") as f:
        f.write("iteration,loss\n")

    enhanced_train_step = decorator(train_step)

    for i in range(100000):
        loss, grad = enhanced_train_step(
            weights=weights,
            calculator=calculator,
            state_vector=state_vector
        )

        opt.apply_gradients(zip(grad, [weights]))

        print(f"loss: {loss}")
        with open("losses.csv", "a+") as f:
            f.write(f"{i},{loss}\n")


if __name__ == "__main__":
    main()
