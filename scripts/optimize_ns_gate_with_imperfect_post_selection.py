import piquasso as pq

import tensorflow as tf

from piquasso._math.fock import cutoff_cardinality


tf.get_logger().setLevel("ERROR")


def _calculate_loss(weights, P, calculator, state_vector, cutoff):
    d = 3
    config = pq.Config(cutoff=cutoff, normalize=False)
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

        pq.ImperfectPostSelectPhotons(
            postselect_modes=(1, 2),
            photon_counts=(1, 0),
            detector_efficiency_matrix=P
        )
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


def train_step(weights, P, calculator, state_vector, cutoff):
    with tf.GradientTape() as tape:
        loss = _calculate_loss(
            weights=weights,
            P=P,
            calculator=calculator,
            state_vector=state_vector,
            cutoff=cutoff
        )

    grad = tape.gradient(loss, weights)

    return loss, grad


def main():
    opt = tf.keras.optimizers.Adam(learning_rate=0.00025)
    decorator = tf.function(jit_compile=True)
    calculator = pq.TensorflowCalculator(decorate_with=decorator)
    np = calculator.np
    fallback_np = calculator.fallback_np
    fallback_np.random.seed(123)

    cutoff = 4

    P = fallback_np.array([
        [1.0, 0.1050, 0.0110, 0.0012, 0.001, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.8950, 0.2452, 0.0513, 0.0097, 0.0017, 0.0003, 0.0001, 0.0],
        [0.0, 0.0, 0.7438, 0.3770, 0.1304, 0.0384, 0.0104, 0.0027, 0.0007],
        [0.0, 0.0, 0.0, 0.5706, 0.4585, 0.2361, 0.0996, 0.0375, 0.0132],
        [0.0, 0.0, 0.0, 0.0, 0.4013, 0.4672, 0.3346, 0.1907, 0.0952],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.2565, 0.4076, 0.3870, 0.2862],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1476, 0.3066, 0.3724],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0755, 0.1985],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9338]
    ])

    P = P[:cutoff, :cutoff]

    ideal_weights = fallback_np.array([np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, - np.pi / 8, 0, 0, 0])
    errors = fallback_np.random.normal(0, 0.1, size=9)
    weights = tf.Variable(ideal_weights, dtype=tf.float64)

    state_vector = np.sqrt([0.2, 0.3, 0.5])

    with open("losses.csv", "a+") as f:
        f.write("iteration,loss\n")

    enhanced_train_step = decorator(train_step)

    for i in range(100000):
        loss, grad = enhanced_train_step(
            weights=weights,
            P=P,
            calculator=calculator,
            state_vector=state_vector,
            cutoff=cutoff
        )

        opt.apply_gradients(zip(grad, [weights]))

        print(f"loss: {loss}")
        with open("losses_imperfect.csv", "a+") as f:
            f.write(f"{i},{loss}\n")


if __name__ == "__main__":
    main()
