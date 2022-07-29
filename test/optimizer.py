import numpy as np
from scipy.optimize import minimize, differential_evolution, dual_annealing, basinhopping, shgo
from scipy import optimize
from reconstruct_manager import ReconstructManager
from tool_box import L2_norm, integrate, compute_e, compute_rho, compute_phi, T_f, get_df_FD_matrix, LP_norm
from projection_manager import ProjectionManager
from equilibrium_manager import EquilibriumManager
from scipy.fftpack import rfft, irfft, rfftfreq


class Optimizer:

    def __init__(self, mesh_x, mesh_v, N):
        self.mesh_x = mesh_x
        self.mesh_v = mesh_v
        self.reconstruct_manager = ReconstructManager(mesh_x, mesh_v, N, scale_coef=False)
        self.projection_manager = ProjectionManager(mesh_x, mesh_v)
        self.N = N
        self.first_optim = True
        self.total_error = 0.
        self.max_error = 0.
        self.nb_it = 0.


    def one_step_optim(self, coef_optim, keys, coef, phi, e, fe, fi, dx_fe, dx_fi, dv_fe, dv_fi):
        mesh_x, mesh_v = self.mesh_x, self.mesh_v

        for i, key in enumerate(keys):
            coef[key] = coef_optim[i]

        eq_manager = EquilibriumManager(mesh_x, mesh_v, coef, init_eq=False)

        exp_a_phi_eq = eq_manager.get_exp_a_phi()
        dx_exp_a_phi_eq = eq_manager.get_dx_exp_a_phi()

        dx_fe_eq, dv_fe_eq = eq_manager.compute_Fe_prime(exp_a_phi_eq, dx_exp_a_phi_eq)
        dx_fi_eq, dv_fi_eq = eq_manager.compute_Fi_prime(exp_a_phi_eq, dx_exp_a_phi_eq)

        fe_eq = eq_manager.compute_Fe(exp_a_phi_eq)
        fi_eq = eq_manager.compute_Fi(exp_a_phi_eq)

        return L2_norm(mesh_x, mesh_v, fe - fe_eq) + L2_norm(mesh_x, mesh_v, fi - fi_eq)

    def optimize_coef(self, fe, fi, eq_manager):
        mesh_x, mesh_v = self.mesh_x, self.mesh_v
        rho = compute_rho(mesh_v, fi - fe)
        phi = compute_phi(mesh_x, -rho)
        A_dx = get_df_FD_matrix(mesh_x.nx, 8)
        A_dv = get_df_FD_matrix(mesh_v.nx, 8)
        dx_fe = np.einsum("ij,jl->il", A_dx, fe) / mesh_x.dx
        dx_fi = np.einsum("ij,jl->il", A_dx, fi) / mesh_x.dx
        dv_fe = np.einsum("jk,lk->jl", fe, A_dv) / mesh_v.dx
        dv_fi = np.einsum("jk,lk->jl", fi, A_dv) / mesh_v.dx
        e = compute_e(mesh_x, rho)
        coef = eq_manager.coef.copy()

        if self.first_optim:
            self.first_optim = False
            self.coef_init = {"x0": coef["x0"]}

        res = minimize(lambda coef_optim: self.one_step_optim(coef_optim, self.coef_init.keys(), coef, phi, e,
         fe, fi, dx_fe, dx_fi, dv_fe, dv_fi), list(self.coef_init.values()), method= 'L-BFGS-B',
            options={'disp': False, 'gtol':1e-12, 'maxiter':1e4})

        self.coef_init = dict(zip(self.coef_init.keys(), res.x))

        for i, key in enumerate(self.coef_init.keys()):
            coef[key] = res.x[i]

        eq_manager_2 = EquilibriumManager(mesh_x, mesh_v, coef)
        fe_eq = eq_manager_2.fe_eq
        fi_eq = eq_manager_2.fi_eq
        rho = compute_rho(mesh_v, fi_eq - fe_eq)
        e_eq = compute_e(mesh_x, rho)
        dx_fe_eq = eq_manager_2.dx_fe_eq
        dx_fi_eq = eq_manager_2.dx_fi_eq
        dv_fe_eq = eq_manager_2.dv_fe_eq
        dv_fi_eq = eq_manager_2.dv_fi_eq
        return fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq, e_eq
