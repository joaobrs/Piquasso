#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from ..circuit import BaseFockCircuit


class FockCircuit(BaseFockCircuit):
    def get_operation_map(self):
        return {
            "DMNumber": self._dm_number,
            **super().get_operation_map(),
        }

    def _dm_number(self, operation):
        self.state._add_occupation_number_basis(
            ket=operation.params[0],
            bra=operation.params[1],
            coefficient=operation.params[2],
        )
