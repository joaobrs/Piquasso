#
# Copyright (C) 2020 by TODO - All rights reserved.
#

import numpy as np


from piquasso.api.state import State
from piquasso.api import constants
from piquasso._math.functions import gaussian_wigner_function

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

    circuit_class = GaussianCircuit

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
    def d(self):
        r"""The number of modes, on which the state is defined.

        Returns:
            int: The number of modes.
        """
        return len(self.m)

    @property
    def xp_mean(self):
        r"""The state's mean in the xp basis.

        The expectation value of the quadrature operators in xp basis, i.e.
        :math:`\operatorname{Tr} \rho \hat{Y}`, where
        :math:`\hat{Y} = (x_1, \dots, x_d, p_1, \dots, p_d)^T`.

        Returns:
            np.array: A :math:`d`-vector.
        """

        _xp_mean = np.empty(2 * self.d)
        _xp_mean[:self.d] = self.m.real * np.sqrt(2 * constants.HBAR)
        _xp_mean[self.d:] = self.m.imag * np.sqrt(2 * constants.HBAR)
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

        return corr * constants.HBAR

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
            np.array:
                The :math:`2d \times 2d` quadrature-ordered covariance matrix in
                xp-ordered basis.
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
            return gaussian_wigner_function(
                quadrature_matrix,
                d=reduced_state.d,
                mean=reduced_state.mu,
                cov=reduced_state.cov
            )

        return gaussian_wigner_function(
            quadrature_matrix,
            d=self.d,
            mean=self.mu,
            cov=self.cov,
        )

    def _apply_passive_linear(self, T, modes):
        r"""Applies the passive transformation `T` to the quantum state.

        See:
            :ref:`passive_gaussian_transformations`

        Args:
            T (numpy.array): The matrix to be applied.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        self.m[modes, ] = T @ self.m[modes, ]

        self._apply_passive_linear_to_C_and_G(T, modes=modes)

    def _apply_passive_linear_to_C_and_G(self, T, modes):
        index = self._get_operator_index(modes)

        self.C[index] = T.conjugate() @ self.C[index] @ T.transpose()
        self.G[index] = T @ self.G[index] @ T.transpose()

        auxiliary_modes = self._get_auxiliary_modes(modes)

        if auxiliary_modes.size != 0:
            self._apply_passive_linear_to_auxiliary_modes(T, modes, auxiliary_modes)

    def _apply_passive_linear_to_auxiliary_modes(self, T, modes, auxiliary_modes):
        auxiliary_index = self._get_auxiliary_operator_index(modes, auxiliary_modes)

        self.C[auxiliary_index] = T.conjugate() @ self.C[auxiliary_index]
        self.G[auxiliary_index] = T @ self.G[auxiliary_index]

        self.C[:, modes] = np.conj(self.C[modes, :]).transpose()
        self.G[:, modes] = self.G[modes, :].transpose()

    def _apply_linear(self, P, A, modes):
        r"""Applies an active transformation to the quantum state.

        See:
            :ref:`active_gaussian_transformations`

        Args:
            P (np.array): A matrix that represents a (P)assive transformation.
            A (np.array): A matrix that represents an (A)ctive transformation.
            modes (tuple): Qumodes on which the transformation directly operates.
        """

        self.m[modes, ] = (
            P @ self.m[modes, ]
            + A @ np.conj(self.m[modes, ])
        )

        self._apply_linear_to_C_and_G(P, A, modes)

    def _apply_linear_to_C_and_G(self, P, A, modes):
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
            self._apply_linear_to_auxiliary_modes(P, A, modes, auxiliary_modes)

    def _apply_linear_to_auxiliary_modes(self, P, A, modes, auxiliary_modes):
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
