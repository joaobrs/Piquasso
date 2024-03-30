import piquasso as pq

import tensorflow as tf

from tqdm import tqdm


tf.get_logger().setLevel("ERROR")


def _calculate_loss(weights, calculator, state_vector, cutoff):
    d = 8
    config = pq.Config(cutoff=cutoff, normalize=False)
    np = calculator.np

    modes = (0, 1, 2, 3)
    ancilla_modes = (4, 5, 6, 7)

    state_00 = [0, 1, 0, 1]
    state_01 = [0, 1, 1, 0]
    state_10 = [1, 0, 0, 1]
    state_11 = [1, 0, 1, 0]

    ancilla_state = [1, 0, 1, 0]

    preparation = pq.Program(instructions=[
        pq.StateVector(state_00 + ancilla_state, coefficient=state_vector[0]),
        pq.StateVector(state_01 + ancilla_state, coefficient=state_vector[1]),
        pq.StateVector(state_10 + ancilla_state, coefficient=state_vector[2]),
        pq.StateVector(state_11 + ancilla_state, coefficient=state_vector[3]),
    ])

    phase_shifter_phis = weights[0][:3]
    thetas = weights[0][3:6]
    phis = weights[0][6:]

    ns_0 = pq.Program(
        instructions=[
            pq.Phaseshifter(phase_shifter_phis[0]).on_modes(modes[0]),
            pq.Phaseshifter(phase_shifter_phis[1]).on_modes(ancilla_modes[0]),
            pq.Phaseshifter(phase_shifter_phis[2]).on_modes(ancilla_modes[1]),

            pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(ancilla_modes[0], ancilla_modes[1]),
            pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(modes[0], ancilla_modes[0]),
            pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(ancilla_modes[0], ancilla_modes[1]),
        ]
    )

    phase_shifter_phis = weights[1][:3]
    thetas = weights[1][3:6]
    phis = weights[1][6:]

    ns_1 = pq.Program(
        instructions=[
            pq.Phaseshifter(phase_shifter_phis[0]).on_modes(modes[2]),
            pq.Phaseshifter(phase_shifter_phis[1]).on_modes(ancilla_modes[2]),
            pq.Phaseshifter(phase_shifter_phis[2]).on_modes(ancilla_modes[3]),

            pq.Beamsplitter(theta=thetas[0], phi=phis[0]).on_modes(ancilla_modes[2], ancilla_modes[3]),
            pq.Beamsplitter(theta=thetas[1], phi=phis[1]).on_modes(modes[2], ancilla_modes[2]),
            pq.Beamsplitter(theta=thetas[2], phi=phis[2]).on_modes(ancilla_modes[2], ancilla_modes[3]),
        ]
    )

    program = pq.Program(instructions=[
        *preparation.instructions,

        pq.Beamsplitter(theta=np.pi / 4).on_modes(0, 2),

        *ns_0.instructions,
        *ns_1.instructions,

        pq.Beamsplitter(theta=-np.pi / 4).on_modes(0, 2),

        pq.PostSelectPhotons(
            postselect_modes=ancilla_modes,
            photon_counts=ancilla_state
        )
    ])

    simulator = pq.PureFockSimulator(d=d, config=config, calculator=calculator)

    state = simulator.execute(program).state
    reduced_state = state.reduced(modes)

    density_matrix = reduced_state.density_matrix

    normalized_density_matrix = density_matrix / tf.linalg.trace(density_matrix)

    expected_program = pq.Program(instructions=[
        pq.StateVector(state_00, coefficient=state_vector[0]),
        pq.StateVector(state_01, coefficient=state_vector[1]),
        pq.StateVector(state_10, coefficient=state_vector[2]),
        pq.StateVector(state_11, coefficient=-state_vector[3]),
    ])

    simulator2 = pq.PureFockSimulator(d=4, config=config, calculator=calculator)
    expected_state = simulator2.execute(expected_program).state
    expected_state_vector = expected_state.state_vector

    loss = 1 - tf.math.real(np.conj(expected_state_vector) @ normalized_density_matrix @ expected_state_vector)

    return loss


def train_step(weights, calculator, state_vector, cutoff):
    with tf.GradientTape() as tape:
        loss = _calculate_loss(
            weights=weights,
            calculator=calculator,
            state_vector=state_vector,
            cutoff=cutoff
        )

    grad = tape.gradient(loss, weights)

    return loss, grad


def main():
    opt = tf.keras.optimizers.Adam(learning_rate=0.00025)
    decorator = tf.function(jit_compile=False)
    calculator = pq.TensorflowCalculator(decorate_with=decorator)
    np = calculator.np
    fallback_np = calculator.fallback_np
    fallback_np.random.seed(123)
    loss_file = "losses_cz_perfect.csv"

    cutoff = 5

    ideal_weights = fallback_np.array([np.pi, 0.0, 0.0, np.pi / 8, 65.5302 * 2 * np.pi / 360, - np.pi / 8, 0, 0, 0])
    errors = fallback_np.random.normal(0, 0.1, size=9)

    weights_np = np.array([
        ideal_weights + errors,
        ideal_weights + errors
    ])
    weights = tf.Variable(weights_np, dtype=tf.float64)

    state_vector = np.sqrt([0.1, 0.2, 0.3, 0.4])

    with open(loss_file, "w") as f:
        f.write("iteration,loss\n")

    enhanced_train_step = decorator(train_step)

    for i in tqdm(range(5000)):
        loss, grad = enhanced_train_step(
            weights=weights,
            calculator=calculator,
            state_vector=state_vector,
            cutoff=cutoff
        )

        opt.apply_gradients(zip([grad], [weights]))

        with open(loss_file, "a+") as f:
            f.write(f"{i},{loss}\n")


if __name__ == "__main__":
    main()
