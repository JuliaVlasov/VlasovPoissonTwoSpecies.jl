import numpy as np
from tool_box import integrate


class ProjectionManager():

    def __init__(self, mesh_x, mesh_v):
        self.mesh_x = mesh_x
        self.mesh_v = mesh_v
        nx          = mesh_x.nx
        nv          = mesh_v.nx
        self.bgk_positions = np.zeros(nx * nv)
        self.bgk_values    = np.zeros(nx * nv)

    def model(self, coef, x):
        return coef[0] * np.exp(-x) + coef[1] * np.exp(-2 * x)

    def model_prime(self, coef, x):
        return - coef[0] * np.exp(-x) -2 * coef[1] * np.exp(-2 * x)

    def update_bgk_arrays(self, f, phi):
        v       = self.mesh_v.x
        nx      = self.mesh_x.nx
        nv      = self.mesh_v.nx

        idx = 0
        for i in range(nx):
            for j in range(nv):
                self.bgk_positions[idx] = phi[i] + v[j]**2 / 2
                self.bgk_values[idx]    = f[i, j]
                idx += 1

        arg_sort           = np.argsort(self.bgk_positions)
        self.bgk_positions = self.bgk_positions[arg_sort]
        self.bgk_values    = self.bgk_values[arg_sort]

    def fit_bgk_function(self, f, phi):
        self.update_bgk_arrays(f, phi)
        dx = self.mesh_x.dx
        dv = self.mesh_v.dx

        model_1_0 = self.model([1, 0], self.bgk_positions)#np.array([self.model([1, 0], xi) for xi in self.bgk_positions])
        model_0_1 = self.model([0, 1], self.bgk_positions)#np.array([self.model([0, 1], xi) for xi in self.bgk_positions])
        #       for idx, xi in enumerate(self.bgk_positions)
        a1 = dx * dv * np.sum(model_1_0**2)
        a2 = dx * dv * np.sum(model_1_0 * model_0_1)
        a3 = dx * dv * np.sum(model_0_1**2)
        b1 = dx * dv * np.sum(model_1_0 * self.bgk_values)
        b2 = dx * dv * np.sum(model_0_1 * self.bgk_values)
        #      for xi in self.bgk_positions])
        # a1 = dx * dv * sum([self.model([1, 0], xi)**2 for xi in self.bgk_positions])
        # a2 = dx * dv * sum([self.model([1, 0], xi) * self.model([0, 1], xi)
        #      for xi in self.bgk_positions])
        # a3 = dx * dv * sum([self.model([0, 1], xi)**2 for xi in self.bgk_positions])
        # b1 = dx * dv * sum([self.model([1, 0], xi) * self.bgk_values[idx]
        #       for idx, xi in enumerate(self.bgk_positions)])
        # b2 = dx * dv * sum([self.model([0, 1], xi) * self.bgk_values[idx]
        #       for idx, xi in enumerate(self.bgk_positions)])

        a11 = a1 # integrate(self.mesh_x, self.mesh_v, a1)
        a12 = a2 # integrate(self.mesh_x, self.mesh_v, a2)
        a22 = a3 # integrate(self.mesh_x, self.mesh_v, a3)

        A     = [[a11, a12], [a12, a22]]
        A_inv = np.linalg.inv(A)
        b = [b1, b2]# [integrate(self.mesh_x, self.mesh_v, b1),
                    # integrate(self.mesh_x, self.mesh_v, b2)]
        b = np.array(b)
        b = b.reshape(b.shape[0], 1)

        return np.matmul(A_inv, b).reshape(-1)

    def project(self, f, phi, coef_only=False):
        coef = self.fit_bgk_function(f, phi)

        if coef_only:
            return coef

        v = self.mesh_v.x

        f_project = [self.model(coef, phi_i + v**2 / 2) for phi_i in phi]

        return np.array(f_project), coef

    def get_df(self, coef, phi, e):
        """
        Return the tuple dx_f, dv_f.
        """
        v = self.mesh_v.x
        f_prime = np.array([self.model_prime(coef, phi_i + v**2 / 2) for phi_i in phi])
        return np.einsum("kj,k->kj", f_prime, -e), np.einsum("kj,j->kj", f_prime, v)


