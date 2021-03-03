#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np

from piquasso import constants, functions
from piquasso.context import Context
from piquasso.state import State

from .circuit import GaussianCircuit

from .transformations import quad_transformation


class GaussianState(State):
    r"""Object to represent a Gaussian state.

    Attributes:
        m (numpy.array): The expectation value of the annihilation operators on all
            modes (a vector, essentially), and is defined by

            .. math::
                m = \langle \hat{a}_i \rangle_{\rho}.

        C (numpy.array): A correlation matrix which is defined by

            .. math::
                \langle \hat{C}_{ij} \rangle_{\rho} =
                \langle \hat{a}^\dagger_i \hat{a}_j \rangle_{\rho}.

        G (numpy.array): A correlation matrix which is defined by

                .. math::
                    \langle \hat{G}_{ij} \rangle_{\rho} =
                    \langle \hat{a}_i \hat{a}_j \rangle_{\rho}.

    """

    _circuit_class = GaussianCircuit

    def __init__(self, C, G, m):
        r"""
        A Gaussian state is fully characterised by its m and correlation
        matrix, i.e. its first and second moments with the quadrature
        operators.

        However, for computations, we only use the
        :math:`C, G \in \mathbb{C}^{d \times d}`
        and the :math:`m \in \mathbb{C}^d` vector.

        Args:
            C (numpy.array): See :attr:`C`.
            G (numpy.array): See :attr:`G`.
            m (numpy.array): See :attr:`m`.

        """

        self.C = C
        self.G = G
        self.m = m

    @classmethod
    def create_vacuum(cls, d):
        r"""Creates a Gaussian vacuum state.

        Args:
            d (int): The number of modes.

        Returns:
            GaussianState: A Gaussan vacuum state.
        """

        return cls(
            C=np.zeros((d, d), dtype=complex),
            G=np.zeros((d, d), dtype=complex),
            m=np.zeros(d, dtype=complex),
        )

    @property
    def hbar(self):
        """Reduced Plack constant.

        TODO: It would be better to move this login into
        :mod:`piquasso.context` after a proper context implementation.

        Returns:
            float: The value of the reduced Planck constant.
        """
        if Context.current_program:
            return Context.current_program.hbar

        return constants.HBAR_DEFAULT

    @property
    def d(self):
        r"""The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return len(self.m)

    @property
    def xp_mean(self):
        r"""The state's mean in the xp basis.

        Returns:
            np.array: A :math:`d`-vector.
                The expectation value of the quadrature operators in xp basis,
                i.e. :math:`\operatorname{Tr} \rho \hat{Y}` , where
                :math:`\hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T`.
        """

        _xp_mean = np.empty(2 * self.d)
        _xp_mean[:self.d] = self.m.real * np.sqrt(2 * self.hbar)
        _xp_mean[self.d:] = self.m.imag * np.sqrt(2 * self.hbar)
        return _xp_mean

    @property
    def xp_corr(self):
        r"""The state's correlation matrix in the xp basis.

        Let :math:`M_{(xp)}` be the correlation matrix in the xp basis.
        Then

        .. math::
            M_{ij (xp)} = \langle Y_i Y_j + Y_j Y_i \rangle_\rho,

        where :math:`M_{ij (xp)}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            \hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` correlation matrix in the xp basis.
        """

        d = self.d

        corr = np.empty((2*d, 2*d), dtype=complex)

        C = self.C
        G = self.G

        corr[:d, :d] = 2 * (G + C).real + np.identity(d)
        corr[:d, d:] = 2 * (G + C).imag
        corr[d:, d:] = 2 * (-G + C).real + np.identity(d)
        corr[d:, :d] = 2 * (G - C).imag

        return corr * self.hbar

    @property
    def xp_cov(self):
        r"""The xp-ordered coveriance matrix of the state.

        The xp-ordered covariance matrix :math:`\sigma_{xp}` is defined by

        .. math::
            \sigma_{xp, ij} := \langle Y_i Y_j + Y_j Y_i \rangle_\rho
                - 2 \langle Y_i \rangle_\rho \langle Y_j \rangle_\rho,

        where

        .. math::
            \hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` xp-ordered covariance matrix in xp basis.
        """
        xp_mean = self.xp_mean
        return self.xp_corr - 2 * np.outer(xp_mean, xp_mean)

    @property
    def xp_representation(self):
        r"""
        The state's mean and correlation matrix ordered in the xp basis.

        Returns:
            tuple: :meth:`xp_mean`, :meth:`xp_corr`.
        """

        return self.xp_mean, self.xp_corr

    @property
    def mu(self):
        r"""Returns the xp-ordered mean of the state.

        Returns:
            np.array: A :math:`2d`-vector.
                The expectation value of the quadrature operators in
                xp-ordering, i.e. :math:`\operatorname{Tr} \rho \hat{R}`, where
                :math:`\hat{R} = (x_1, p_1, \dots, x_d, p_d)^T`.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_mean

    @property
    def corr(self):
        r"""Returns the quadrature-ordered correlation matrix of the state.

        Let :math:`M` be the correlation matrix in the quadrature basis.
        Then

        .. math::
            M_{ij} = \langle R_i R_j + R_j R_i \rangle_\rho,

        where :math:`M_{ij}` denotes the matrix element in the
        :math:`i`-th row and :math:`j`-th column,

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` quad-ordered correlation matrix.
        """
        T = quad_transformation(self.d)
        return T @ self.xp_corr @ T.transpose()

    @property
    def cov(self):
        r"""The quadrature-ordered coveriance matrix of the state.

        The quadrature-ordered covariance matrix :math:`\sigma` is defined by

        .. math::
            \sigma_{ij} := \langle R_i R_j + R_j R_i \rangle_\rho
                - 2 \langle R_i \rangle_\rho \langle R_j \rangle_\rho,

        where

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\rho` is the density operator of the currently represented state.

        Returns:
            np.array: The :math:`2d \times 2d` quadrature-ordered covariance matrix in
                xp basis.
        """
        mu = self.mu
        return self.corr - 2 * np.outer(mu, mu)

    @property
    def quad_representation(self):
        r"""The state's mean and correlation matrix ordered by the quadrature basis.

        Returns:
            tuple: :meth:`mu`, :meth:`corr`.
        """

        return self.mu, self.corr

    def rotated(self, phi):
        r"""Returns the copy of the current state, rotated by `phi`.

        Let :math:`\phi \in [ 0, 2 \pi )`. Let us define the following:

        .. math::
            x_{i, \phi} = \cos\phi~x_i + \sin\phi~p_i,

        which is a generalized quadrature operator. One could rotate the whole state by
        this simple, phase space transformation.

        Using the transformation rules between the ladder operators and quadrature
        operators, i.e.

        .. math::
            x_i &= \sqrt{\frac{\hbar}{2}} (a_i + a_i^\dagger) \\
            p_i &= -i \sqrt{\frac{\hbar}{2}} (a_i - a_i^\dagger),

        we could rewrite :math:`x_{i, \phi}` to the following form:

        .. math::
            x_{i, \phi} = \sqrt{\frac{\hbar}{2}} \left(
                a_i \exp(-i \phi) + a_i^\dagger \exp(i \phi)
            \right)

        which means, that e.g. the annihilation operators `a_i` are transformed just
        multiplied by a phase factor :math:`\exp(-i \phi)` under this phase space
        rotation, i.e.

        .. math::
            (\langle a_i \rangle_{\rho} =: )~m_i &\mapsto \exp(-i \phi) m_i \\
            (\langle a^\dagger_i a_j \rangle_{\rho} =: )~C_{ij} &\mapsto C_{ij} \\
            (\langle a_i a_j \rangle_{\rho} =: )~G_{ij} &\mapsto \exp(-i 2 \phi) G_{ij}.

        Args:
            phi (float): The angle to rotate the state with.

        Returns:
            GaussianState: The rotated `GaussianState` instance.
        """
        phase = np.exp(- 1j * phi)

        return GaussianState(
            C=self.C,
            G=(self.G * phase**2),
            m=(self.m * phase),
        )

    def reduced(self, modes):
        """Returns the copy of the current state, reduced to the given `modes`.

        This method essentially preserves the modes specified from the representation
        of the Gaussian state, but cuts out the other modes.

        Args:
            modes (tuple): The modes to reduce the state to.

        Returns:
            GaussianState: The reduced `GaussianState` instance.
        """
        return GaussianState(
            C=self.C[np.ix_(modes, modes)],
            G=self.G[np.ix_(modes, modes)],
            m=self.m[np.ix_(modes)],
        )

    def reduced_rotated_mean_and_cov(self, modes, phi):
        r"""The quadrature operator's mean and covariance on a rotated and reduced state.

        Let the index set :math:`\vec{i}` correspond to `modes`, and the angle
        :math:`\phi` correspond to `phi`. The current :class:`GaussianState` instance
        is reduced to `modes` and rotated by `phi` in a new instance, and let that
        state be denoted by :math:`\rho_{\vec{i}, \phi}`.

        Then the quadrature ordered mean and covariance can be calculated by

        .. math::
            \mu_{\vec{i}, \phi}
                &:= \langle \hat{R}_{\vec{i}} \rangle_{\rho_{\vec{i}, \phi}}, \\
            \sigma_{\vec{i}, \phi}
                &:=  \langle
                    \hat{R}_{\vec{i}} \hat{R}_{\vec{i}}^T
                \rangle_{\rho_{\vec{i}, \phi}}
                - \mu_{\vec{i}, \phi} \mu_{\vec{i}, \phi}^T,

        where

        .. math::
            \hat{R} = (x_1, p_1, \dots, x_d, p_d)^T,

        and :math:`\hat{R}_{\vec{i}}` is just the same vector, reduced to a subsystem
        specified by :math:`\vec{i}`.

        Args:
            modes (tuple): The modes to reduce the state to.
            phi (float): The angle to rotate the state with.

        Returns:
            tuple:
                Quadrature ordered mean and covariance of the reduced and rotated
                version of the current :class:`GaussianState`.
        """
        transformed_state = self.reduced(modes).rotated(phi)

        return transformed_state.mu, transformed_state.cov

    def wigner_function(self, quadrature_matrix, modes=None):
        r"""
        Calculates the Wigner function values at the specified `quadrature_matrix`,
        according to the equation

        .. math::
            W(r) = \frac{1}{\pi^d \sqrt{\mathrm{det} \sigma}}
                \exp \big (
                    - (r - \mu)^T
                    \sigma^{-1}
                    (r - \mu)
                \big ).

        Args:
            quadrature_matrix (list): list of canonical coordinates vectors.
            modes (tuple or None): modes where Wigner function should be calculcated.

        Returns:
            tuple: The Wigner function values in the shape of `quadrature_matrix`.
        """

        if modes:
            reduced_state = self.reduced(modes)
            return functions.gaussian_wigner_function(
                quadrature_matrix,
                d=reduced_state.d,
                mean=reduced_state.mu,
                cov=reduced_state.cov
            )

        return functions.gaussian_wigner_function(
            quadrature_matrix,
            d=self.d,
            mean=self.mu,
            cov=self.cov,
        )

    def apply_passive(self, T, modes):
        r"""Applies a passive transformation to the quantum state.

        Let :math:`\vec{m}` denote an index set, which corresponds to the parameter
        `modes`.

        Let :math:`T \in \mathbb{C}^{k \times k},\, k \in [d]` be a transformation
        which transforms the vector of annihilation operators in the following manner:

        .. math::
            \mathbf{a}_{\vec{m}} \mapsto T \mathbf{a}_{\vec{m}},

        or in terms of vector elements:

        .. math::
            a_{i} \mapsto \sum_{j \in \vec{m}} T^{ij} a_j

        Application to :attr:`m` is done by matrix multiplication.

        The canonical commutation relations can be written as

        .. math::
            [a^\dagger_i, a_j] = \delta_{i j},

        and then applying the transformation :math:`T` we get

        .. math::
            \sum_{i, j \in \vec{m}} [T^*_{ki} a^\dagger_i, T_{lj} a_j]
                &= \sum_{i, j \in \vec{m}} T^*_{ki} T_{lj}
                    [a^\dagger_i, a_j] \\
                &= \sum_{i, j \in \vec{m}} T^*_{ki} T_{lj} \delta_{i j} \\
                &= \sum_{i \in \vec{m}} T^*_{ki} T_{li} \\
                &= \sum_{i \in \vec{m}} (T^\dagger)_{ik} T_{li} \\
                &= \delta_{k l},

        where the last line imposes, that any transformation should leave the canonical
        commutation relations invariant.
        The last line of the equation means, that :math:`T` should actually be a
        unitary matrix.

        Application to `C` and `G` is non-trivial however: one has to apply the
        transformation for the external modes as well, see
        :meth:`_apply_passive_to_C_and_G`.

        Args:
            T (numpy.array): The matrix to be applied.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        self.m[modes, ] = T @ self.m[modes, ]

        self._apply_passive_to_C_and_G(T, modes=modes)

    def _apply_passive_to_C_and_G(self, T, modes):
        r"""Applies the transformation :math:`T` to the :math:`C` and :math:`G`.

        Let :math:`\vec{i}` denote an index set, which corresponds to `index`
        in the implementation. E.g. for 2 modes denoted by :math:`n` and
        :math:`m`:, one could write

        .. math::
            \vec{i} = \{n, m\} \times \{n, m\}.

        From now on, I will use the notation
        :math:`\{n, m\} := \mathrm{modes}`.

        The transformation by :math:`T` can be prescribed in the following
        manner:

        .. math::
                C_{\vec{i}} \mapsto T^* C_{\vec{i}} T^T \\
                G_{\vec{i}} \mapsto T G_{\vec{i}} T^T

        If there are other modes in the system, i.e. `modes` does not refer to all the
        modes, :meth:`_apply_passive_to_auxiliary_modes` is called to handle those.

        Note:
            For indexing of numpy arrays, see
            https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing

        Args:
            T (np.array): The matrix to be applied.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        index = self._get_operator_index(modes)

        self.C[index] = T.conjugate() @ self.C[index] @ T.transpose()
        self.G[index] = T @ self.G[index] @ T.transpose()

        auxiliary_modes = self._get_auxiliary_modes(modes)

        if auxiliary_modes.size != 0:
            self._apply_passive_to_auxiliary_modes(T, modes, auxiliary_modes)

    def _apply_passive_to_auxiliary_modes(self, T, modes, auxiliary_modes):
        r"""Applies the matrix :math:`T` to modes which are not directly transformed.

        This method is applied for the correlation matrices :math:`C` and :math:`G`.
        For context, visit :meth:`_apply_passive_to_C_and_G`.

        Let us denote :math:`\vec{k}` the following:

        .. math::
                \vec{k} = \mathrm{modes}
                        \times \big (
                                [d]
                                - \mathrm{modes}
                        \big ).

        For all the remaining modes, the following is applied regarding the
        elements, where the **first** index corresponds to
        :math:`\mathrm{modes}`:

        .. math::
                C_{\vec{k}} \mapsto T^* C_{\vec{k}} \\
                G_{\vec{k}} \mapsto T G_{\vec{k}}

        Regarding the case where the **second** index corresponds to
        :math:`\mathrm{modes}`, i.e. where we use
        :math:`\big ( [d] - \mathrm{modes} \big )
        \times \mathrm{modes}`, the same has to be applied.

        For :math:`n \in \mathrm{modes}` and :math:`m \in [d]`, we could
        use

        .. math::
                C_{nm} := C^*_{mn} \\
                G_{nm} := G_{mn}.


        Args:
            T (np.array): The matrix to be applied.
            modes (tuple): Qumodes on which the transformation directly operates.
            auxiliary_modes (tuple):
                The modes, on which the transformation is not directly applied, but
                should be accounted for in :math:`C` and :math:`G`.
        """

        auxiliary_index = self._get_auxiliary_operator_index(modes, auxiliary_modes)

        self.C[auxiliary_index] = T.conjugate() @ self.C[auxiliary_index]
        self.G[auxiliary_index] = T @ self.G[auxiliary_index]

        self.C[:, modes] = np.conj(self.C[modes, :]).transpose()
        self.G[:, modes] = self.G[modes, :].transpose()

    def apply_active(self, P, A, modes):
        r"""Applies an active transformation to the quantum state.

        Let :math:`\vec{m}` denote an index set, which corresponds to the parameter
        `modes`.

        Let :math:`P, A \in \mathbb{C}^{k \times k},\, k \in [d]` be a passive and an
        active transformation, respectively. An active operation transforms the vector
        of annihilation operators in the following manner:

        .. math::
             \mathbf{a}_{\vec{m}}
                P \mathbf{a}_{\vec{m}} + A \mathbf{a}_{\vec{m}^*},

        or in terms of vector elements:

        .. math::
            a_{i} \mapsto
                \sum_{j \in \vec{m}} P^{ij} a_j
                + \sum_{j \in \vec{m}} A^{ij} a_j^\dagger

        The vector of the means of the :math:`i`-th mode
        :math:`m_i = \langle \hat{a}_i \rangle_\rho` is evolved as follows:

        .. math::
            m_i \mapsto P m_i + A m_i^*

        Args:
            P (np.array): A matrix that represents a (P)assive transformation.
            A (np.array): A matrix that represents an (A)assive transformation.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        self.m[modes, ] = (
            P @ self.m[modes, ]
            + A @ np.conj(self.m[modes, ])
        )

        self._apply_active_to_C_and_G(P, A, modes)

    def _apply_active_to_C_and_G(self, P, A, modes):
        r"""Applies an active transformation to the C and G matrices.

        The transformations in the terms of the transformation matrices are defined by

        .. math::
            G_{i j} \mapsto
                (P G P^T + A G^\dagger A^T + P (1 + C^T) A^T + A C P^T)_{i j}

        .. math::
            C_{i j} \mapsto
                (P^* C P^T + A^* (1 + C^T) P^T + P^* G^\dagger A^T + A^* G P^T)_{i j}

        If there are other modes in the system, i.e. `modes` does not refer to all the
        modes, :meth:`_apply_active_to_auxiliary_modes` is called to handle those.

        Args:
            P (np.array): A matrix that represents a (P)assive transformation.
            A (np.array): A matrix that represents an (A)assive transformation.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        index = self._get_operator_index(modes)

        original_C = self.C[index]
        original_G = self.G[index]

        self.G[index] = (
            P @ original_G @ P.transpose()
            + A @ original_G.conjugate().transpose() @ A.transpose()
            + P @ (original_C.transpose() + np.identity(len(modes))) @ A.transpose()
            + A @ original_C @ P.transpose()
        )

        self.C[index] = (
            P.conjugate() @ original_C @ P.conjugate().transpose()
            + A.conjugate() @ (
                original_C.transpose() + np.identity(len(modes))
            ) @ A.transpose()
            + P.conjugate() @ original_G.conjugate().transpose() @ A.transpose()
            + A.conjugate() @ original_G @ P.transpose()
        )

        auxiliary_modes = self._get_auxiliary_modes(modes)

        if auxiliary_modes.size != 0:
            self._apply_active_to_auxiliary_modes(P, A, modes, auxiliary_modes)

    def _apply_active_to_auxiliary_modes(self, P, A, modes, auxiliary_modes):
        r"""
        This method updates the off diagonal elements of the :math:`G` and the :math:`C`
        matrices.

        The columns :math:`j` defined in `auxiliary_modes` associated with the mode
        :math:`i` defined in `modes` evolve according to the linear transformation
        defined in :meth:`apply_active`.

        Then each row of the mode :math:`i` will be updated according to the fact that
        :math:`C_{ij} = C_{ij}^*` and :math:`G_{ij} = G_{ij}^T`.

        Args:
            alpha (complex): A complex that represents the value of :math:`cosh(amp)`.
            beta (complex):
                A complex that represents the value of :math:`e^{i\theta}\sinh(amp)`.
            modes (tuple): Qumodes on which the transformation directly operates.
            auxiliary_modes (np.array): A vector that contains The modes, on which the
                transformation is not directly applied, but should be accounted for in
                :math:`C` and :math:`G`.
        """
        auxiliary_index = self._get_auxiliary_operator_index(modes, auxiliary_modes)

        auxiliary_C = self.C[auxiliary_index]
        auxiliary_G = self.G[auxiliary_index]

        self.C[auxiliary_index] = (
            P.conjugate() @ auxiliary_C
            + A.conjugate() @ auxiliary_G
        )

        self.G[auxiliary_index] = (
            P @ auxiliary_G
            + A @ auxiliary_C
        )

        self.C[:, modes] = self.C[modes, :].conjugate().transpose()
        self.G[:, modes] = self.G[modes, :].transpose()
