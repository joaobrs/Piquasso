#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

import piquasso as pq


def test_measure_particle_number_on_one_mode():
    with pq.Program() as program:
        pq.Q() | pq.FockState(d=3, cutoff=2)

        pq.Q() | pq.Number(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.Number(ket=(0, 0, 2), bra=(1, 0, 1)) * (- 3j)

        pq.Q() | pq.Number(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(1, 0, 1)) * (- 1j)

        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.Number(ket=(0, 1, 1), bra=(0, 0, 1)) * (- 2j)

        pq.Q() | pq.Number(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.Number(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (1, ) or outcome == (2, )

    if outcome == (1, ):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                1/3 * pq.Number(ket=(0, 0, 1), bra=(0, 0, 1)),
                4j * pq.Number(ket=(0, 0, 1), bra=(0, 1, 1)),
                -2j * pq.Number(ket=(0, 0, 1), bra=(1, 0, 1)),
                -4j * pq.Number(ket=(0, 1, 1), bra=(0, 0, 1)),
                2 / 3 * pq.Number(ket=(0, 1, 1), bra=(0, 1, 1)),
                2j * pq.Number(ket=(1, 0, 1), bra=(0, 0, 1)),
            ]
        )

    elif outcome == (2, ):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(0, 0, 2), bra=(0, 0, 2))
            ]
        )

    assert program.state == expected_state


def test_measure_particle_number_on_two_modes():
    with pq.Program() as program:
        pq.Q() | pq.FockState(d=3, cutoff=2)

        pq.Q() | pq.Number(ket=(1, 0, 1), bra=(0, 0, 2)) * 3j
        pq.Q() | pq.Number(ket=(0, 0, 2), bra=(1, 0, 1)) * (- 3j)

        pq.Q() | pq.Number(ket=(1, 0, 1), bra=(0, 0, 1)) * 1j
        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(1, 0, 1)) * (- 1j)

        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(0, 1, 1)) * 2j
        pq.Q() | pq.Number(ket=(0, 1, 1), bra=(0, 0, 1)) * (- 2j)

        pq.Q() | pq.Number(ket=(0, 1, 1), bra=(0, 1, 1)) * 2 / 6
        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(0, 0, 1)) * 1 / 6
        pq.Q() | pq.Number(ket=(0, 0, 2), bra=(0, 0, 2)) * 3 / 6

        pq.Q(1, 2) | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (0, 1) or outcome == (1, 1) or outcome == (0, 2)

    if outcome == (0, 1):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(0, 0, 1), bra=(0, 0, 1)),
                pq.Number(ket=(0, 0, 1), bra=(1, 0, 1)) * (-6j),
                pq.Number(ket=(1, 0, 1), bra=(0, 0, 1)) * 6j,
            ]
        )

    elif outcome == (1, 1):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(0, 1, 1), bra=(0, 1, 1)),
            ]
        )

    elif outcome == (0, 2):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(0, 0, 2), bra=(0, 0, 2))
            ]
        )

    assert program.state == expected_state


def test_measure_particle_number_on_all_modes():
    with pq.Program() as preparation:
        pq.Q() | pq.FockState(d=3, cutoff=2)

        pq.Q() | pq.Number(ket=(0, 0, 0), bra=(0, 0, 0)) / 4

        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(0, 0, 1)) / 4
        pq.Q() | pq.Number(ket=(1, 0, 0), bra=(1, 0, 0)) / 2

        pq.Q() | pq.Number(ket=(0, 0, 1), bra=(1, 0, 0)) * np.sqrt(1/8)
        pq.Q() | pq.Number(ket=(1, 0, 0), bra=(0, 0, 1)) * np.sqrt(1/8)

    with pq.Program() as program:
        pq.Q() | preparation

        pq.Q() | pq.MeasureParticleNumber()

    results = program.execute()

    assert np.isclose(sum(program.state.fock_probabilities), 1)
    assert len(results) == 1

    outcome = results[0].outcome
    assert outcome == (0, 0, 0) or outcome == (0, 0, 1) or outcome == (1, 0, 0)

    if outcome == (0, 0, 0):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(0, 0, 0), bra=(0, 0, 0)),
            ]
        )

    elif outcome == (0, 0, 1):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(0, 0, 1), bra=(0, 0, 1)),
            ]
        )

    elif outcome == (1, 0, 0):
        expected_state = pq.FockState.from_number_preparations(
            d=3, cutoff=2,
            number_preparations=[
                pq.Number(ket=(1, 0, 0), bra=(1, 0, 0)),
            ]
        )

    assert program.state == expected_state
