#
# Copyright 2021 Budapest Quantum Computing Group
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
import inspect
from collections import OrderedDict
from typing import List, Mapping, Optional, Type

import blackbird as bb

from . import _registry
from .. import Instruction
from ..api.errors import PiquassoException


def load_instructions(blackbird_program: bb.BlackbirdProgram) -> List[Instruction]:
    """
    Loads the gates to apply into :attr:`Program.instructions` from a
    :class:`~blackbird.program.BlackbirdProgram`.

    Args:
        blackbird_program (~blackbird.program.BlackbirdProgram):
            The Blackbird program to use.
    """

    instruction_map = {
        "Dgate": _registry.items["Displacement"],
        "Xgate": _registry.items["PositionDisplacement"],
        "Zgate": _registry.items["MomentumDisplacement"],
        "Sgate": _registry.items["Squeezing"],
        "Pgate": _registry.items["QuadraticPhase"],
        "Vgate": None,
        "Kgate": _registry.items["Kerr"],
        "Rgate": _registry.items["Phaseshifter"],
        "BSgate": _registry.items["Beamsplitter"],
        "MZgate": _registry.items["MachZehnder"],
        "S2gate": _registry.items["Squeezing2"],
        "CXgate": _registry.items["ControlledX"],
        "CZgate": _registry.items["ControlledZ"],
        "CKgate": _registry.items["CrossKerr"],
        "Fouriergate": _registry.items["Fourier"],
    }

    return [
        _blackbird_operation_to_instruction(instruction_map, operation)
        for operation in blackbird_program.operations
    ]


def _blackbird_operation_to_instruction(
    instruction_map: Mapping[str, Optional[Type[Instruction]]],
    blackbird_operation: dict
) -> Instruction:
    op = blackbird_operation["op"]
    pq_instruction_class = instruction_map.get(op)

    if pq_instruction_class is None:
        raise PiquassoException(
            f"Operation {op} is not implemented in piquasso."
        )

    params = _get_instruction_params(
        pq_instruction_class=pq_instruction_class, bb_operation=blackbird_operation
    )

    instruction = pq_instruction_class(**params)

    instruction.modes = tuple(blackbird_operation["modes"])

    return instruction


def _get_instruction_params(
    pq_instruction_class: Type[Instruction], bb_operation: dict
) -> dict:
    bb_params = bb_operation.get("args", None)

    if bb_params is None:
        return {}

    parameters = inspect.signature(pq_instruction_class).parameters

    instruction_params = OrderedDict()

    for param_name, param in parameters.items():
        if param_name == "self":
            continue

        instruction_params[param_name] = param.default

    for pq_param_name, bb_param in zip(instruction_params.keys(), bb_params):
        instruction_params[pq_param_name] = bb_param

    return instruction_params
