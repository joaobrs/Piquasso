#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from piquasso.core.mixins import _PropertyMixin, _RegisterMixin


class Instruction(_PropertyMixin, _RegisterMixin):
    """
    Args:
        *params: Variable length argument list.
    """

    def __init__(self, **params):
        self._set_params(**params)

    def _set_params(self, **params):
        self.params = params

    def on_modes(self, *modes):
        self.modes = modes
        return self

    def apply_to_program_on_register(self, program, register):
        program.instructions.append(self.on_modes(*register.modes))

    @classmethod
    def from_properties(cls, properties):
        """Creates an `Instruction` instance from a mapping specified.

        Args:
            properties (collections.Mapping):
                The desired `Operator` instance in the format of a mapping.

        Returns:
            Operator: An `Operator` initialized using the specified mapping.
        """

        instruction = cls(**properties["params"])

        instruction.modes = properties["modes"]

        return instruction

    def __repr__(self):
        if hasattr(self, "modes"):
            modes = "modes={}".format(self.modes)
        else:
            modes = ""

        if hasattr(self, "params"):
            params = "{}".format(
                ", ".join(
                    [f"{key}={value}" for key, value in self.params.items()]
                )
            )
        else:
            params = ""

        classname = self.__class__.__name__

        return f"<pq.{classname}({params}, {modes})>"

    def __eq__(self, other):
        return (
            self.modes == other.modes
            and
            self.params == other.params
        )
