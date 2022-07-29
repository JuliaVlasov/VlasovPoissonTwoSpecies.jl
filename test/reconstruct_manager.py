import numpy as np
from tool_box import compute_rho, compute_e, T_f, get_df_FD_matrix, integrate, L2_norm
from scipy.fftpack import rfft

class ReconstructManager:

    def __init__(self, mesh_x, mesh_v, N, scale_coef=True):
        self.N = N
        self.mesh_x = mesh_x
        self.mesh_v = mesh_v
        self.fourier_series = []
        self.scale_coef = scale_coef
        if not scale_coef:
            self.length_coef = 2 * (N) + 3
        else:
            self.length_coef = 2 * (N) + 2
        self.coef_exp1_threshold = 0.1


        nx = mesh_x.nx
        nv = mesh_v.nx

        self.A_dx = get_df_FD_matrix(nx)
        self.A_dv = get_df_FD_matrix(nv)

        # Create the constant function
        self.fourier_series.append(np.fft.ifft([nx], nx).real)

        for i in range(0, N):
            idx = i + 1
            coef = np.zeros(nx, dtype=complex)
            coef[idx] = nx
            self.fourier_series.append(np.fft.ifft(coef,nx).real)

            coef = np.zeros(nx, dtype=complex)
            coef[idx] = - nx * 1j
            self.fourier_series.append(np.fft.ifft(coef,nx).real)

        self.fourier_series = np.array(self.fourier_series)
        v = mesh_v.x
        self.velocity_diag1 = np.array(
            [np.exp(-(vj**2) / 2) for vj in v])
        self.velocity_diag2 = np.array(
            [np.exp(-vj**2) for vj in v])

    def get_fourier_coef(self, f):
        mesh_x = self.mesh_x
        length = (mesh_x.x_max - mesh_x.x_min)
        f_fft_coef = 2 * mesh_x.dx * rfft(f, mesh_x.nx) / length
        f_fft_coef = f_fft_coef[:1 + 2 * self.N]
        f_fft_coef[0] /= 2
        for i in range((f_fft_coef.shape[0] - 1) // 2):
            idx = 2 * (i + 1)
            f_fft_coef[idx] *= -1

        return f_fft_coef

    def reconstruct_fourier(self, coef):
        return np.einsum("k,kj->j", coef, self.fourier_series)

    def reconstruct_scale_function(self, coef_integral):
        coef = coef_integral[0]
        integral = coef_integral[1]

        # , integral=integral)
        return self.reconstruct_function(coef, integral)

    def reconstruct_function(self, coef, integrals=None):
        """
        Reconstruct the function of the form:
        coef[0]*exp(-2*(v**2/2+fourier))+coef[1]*exp(-(v**2/2+fourier)),
        where fourier is the fourier series reconstruct with the coefficients
        coef[2:]
        """

        # If scale_fourier = True, the first fourier coef is one by default,
        # coef_exp bounded and we rescale the fourier series so that
        # min(fourier)>=0
        assert(coef.shape[0] == self.length_coef and len(coef.shape) == 1), \
            f"Can't reconstruct function with coef with shape {coef.shape}"

        coef_exp1 = coef[0]
        coef_exp2 = coef[1]

        if self.scale_coef:
            fourier_coef = np.ones(coef[2:].shape[0] + 1)
            fourier_coef[1:] = coef[2:]
        else:
            fourier_coef = coef[2:]

        fourier      = np.einsum("k,kj->j", fourier_coef, self.fourier_series)
        fourier_min  = np.amin(fourier)
        if self.scale_coef:
            fourier      -= fourier_min
            fourier_min  = 0.

        fourier_mat1 = np.exp(-fourier)#fourier#
        # tf.math.exp(tf.stack([fourier]*nv, axis=1))
        fourier_mat2 = np.exp(-2 * fourier)#fourier#
        # fourier_mat = tf.stack([fourier]*nv, axis=1)

        coef_exp1 = np.maximum(coef_exp1, self.coef_exp1_threshold)
        coef_exp2 = np.maximum(coef_exp2, -coef_exp1 * np.exp(fourier_min))
        velocity_mat1 = coef_exp1 * self.velocity_diag1
        velocity_mat2 = coef_exp2 * self.velocity_diag2

        f = np.einsum("i,j->ij", fourier_mat1, velocity_mat1) + np.einsum("i,j->ij", fourier_mat2, velocity_mat2)

        # Normalize the integrals.
        if integrals is not None:
            assert len(integrals.shape) in (0, 1), \
                f"Can't use integrals of shape {integrals.shape}"

            f_integrals = integrate(self.mesh_x, self.mesh_v, f)
            f *= integrals / f_integrals

        return f

    def error(self, coef, f2, positive_sign=True):

        mesh_x = self.mesh_x
        mesh_v = self.mesh_v
        nx = mesh_x.nx
        nv = mesh_v.nx

        assert(len(coef.shape) == 1 and coef.shape[0] == self.length_coef), \
            f"Can't calculate the error with coef of shape {coef.shape}"
        assert((f2.shape[0], f2.shape[1]) == (nx, nv) and len(f2.shape) == 2), \
            f"Can't calculate the error of f2 with shape {f2.shape}"

        integrals = integrate(mesh_x, mesh_v, f2)
        f = self.reconstruct_function(coef, integrals=integrals)

        assert((f.shape[0], f.shape[1]) == (nx, nv) and len(f.shape) == 2), \
            f"Can't calculate the error of f with shape {f.shape}"

        coef_sign = 1
        if not positive_sign:
            coef_sign = -1

        rho = compute_rho(mesh_v, coef_sign * (f - f2))
        e = compute_e(mesh_x, rho)

        T_f_f2 = T_f(mesh_x, mesh_v, f, coef_sign * e, A_dx=self.A_dx, A_dv=self.A_dv) + T_f(mesh_x, mesh_v, f2, -coef_sign * e, A_dx=self.A_dx, A_dv=self.A_dv)

        error = L2_norm(mesh_x, mesh_v, T_f_f2) / integrals

        return error
