import numpy as np
from advection import Advection
from tool_box import compute_rho, compute_e, T_f, integrate, compute_phi
from projection_manager import ProjectionManager
from optimizer import Optimizer

def advect_vlasov(advection_x, advection_v, fe, fi, dt, e_eq=None, order=2):

    mesh_x = advection_x.mesh
    mesh_v = advection_v.mesh

    v = mesh_v.x

    if order == 1:

        rho = compute_rho(mesh_v, fi - fe)
        e   = compute_e(mesh_x, rho)

        # In case of wb scheme we add e_eq to e.
        if e_eq is not None:
            e += e_eq

        advection_v.advect(fe, -e, dt)
        advection_v.advect(fi, e, dt)

        advection_x.advect(np.transpose(fe), v, dt)
        advection_x.advect(np.transpose(fi), v, dt)
    elif order == 2:
        advection_x.advect(np.transpose(fe), v, 0.5 * dt)
        advection_x.advect(np.transpose(fi), v, 0.5 * dt)

        rho = compute_rho(mesh_v, fi - fe)
        e   = compute_e(mesh_x, rho)

        # In case of wb scheme we add e_eq to e.
        if e_eq is not None:
            e += e_eq

        advection_v.advect(fe, -e, dt)
        advection_v.advect(fi, e, dt)

        advection_x.advect(np.transpose(fe), v, 0.5 * dt)
        advection_x.advect(np.transpose(fi), v, 0.5 * dt)
    else:
        raise ValueError(f"No advection of order {order}.")


class Scheme():

    def __init__(self, mesh_x, mesh_v, fe, fi, fe_eq=None, fi_eq=None,
    dx_fe_eq=None, dx_fi_eq=None, dv_fe_eq=None, dv_fi_eq=None, wb_scheme=False):
        self.mesh_x = mesh_x
        self.mesh_v = mesh_v
        self.advection_x = Advection(mesh_x)
        self.advection_v = Advection(mesh_v)
        self.fe = fe
        self.fi = fi
        self.fe_eq = fe_eq
        self.fi_eq = fi_eq
        self.dx_fe_eq = dx_fe_eq
        self.dx_fi_eq = dx_fi_eq
        self.dv_fe_eq = dv_fe_eq
        self.dv_fi_eq = dv_fi_eq
        self.wb_scheme = wb_scheme
        self.e_eq = None
        self.projection_manager = ProjectionManager(mesh_x, mesh_v)
        self.e_projection = None
        self.optimizer = Optimizer(mesh_x, mesh_v, 15)
        self.coef_fe_init = None
        self.coef_fi_init = None
        self.coef_init = None

        if wb_scheme:
            assert fe_eq is not None and fi_eq is not None, \
                f"Error: wb scheme is used but equilibriums are not provided."
            rho_eq    = compute_rho(mesh_v, fi_eq - fe_eq)
            self.e_eq = compute_e(mesh_x, rho_eq)
            self.ge   = fe - fe_eq
            self.gi   = fi - fi_eq

    def compute_wb_source(self, dt):
        mesh_x, mesh_v, ge, gi, dv_fe_eq, dv_fi_eq = (self.mesh_x, self.mesh_v,
            self.ge, self.gi, self.dv_fe_eq, self.dv_fi_eq)
        rho_g = compute_rho(mesh_v, gi - ge)
        e_g   = compute_e(mesh_x, rho_g)
        for i in range(ge.shape[0]):
            self.ge[i, :] += dt * e_g[i] * dv_fe_eq[i, :]
        for i in range(gi.shape[0]):
            self.gi[i, :] -= dt * e_g[i] * dv_fi_eq[i, :]

    def compute_wb_vlasov(self, dt):
        advection_x, advection_v, e_eq = (self.advection_x,
            self.advection_v, self.e_eq)

        self.compute_wb_source(0.5 * dt)
        advect_vlasov(advection_x, advection_v, self.ge, self.gi, dt, e_eq, order=2)
        self.compute_wb_source(0.5 * dt)
        self.fe = self.fe_eq + self.ge
        self.fi = self.fi_eq + self.gi

    def compute_wb_source_2(self, dt):
        (mesh_x, mesh_v, ge, gi, fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq,
            dv_fi_eq) = (self.mesh_x, self.mesh_v, self.ge, self.gi, self.fe_eq,
            self.fi_eq, self.dx_fe_eq, self.dx_fi_eq, self.dv_fe_eq, self.dv_fi_eq)
        e_projection = self.e_projection

        rho = compute_rho(mesh_v, (fi_eq + gi) - (fe_eq + ge))
        e   = compute_e(mesh_x, rho)
        if True:#e_projection is None:
            T_fe_eq = T_f(mesh_x, mesh_v, fe_eq, -e, dx_f=dx_fe_eq, dv_f=dv_fe_eq)
            T_fi_eq = T_f(mesh_x, mesh_v, fi_eq, e, dx_f=dx_fi_eq, dv_f=dv_fi_eq)
            self.ge[:, :] -= dt * T_fe_eq[:, :]
            self.gi[:, :] -= dt * T_fi_eq[:, :]
        else:
            for i in range(ge.shape[0]):
                self.ge[i, :] += dt * (e[i] - e_projection[i]) * dv_fe_eq[i, :]
                self.gi[i, :] -= dt * (e[i] - e_projection[i]) * dv_fi_eq[i, :]

    def compute_wb_vlasov_2(self, dt):
        (advection_x, advection_v, e_eq) = (self.advection_x,
            self.advection_v, self.e_eq)

        self.compute_wb_source_2(0.5 * dt)
        advect_vlasov(advection_x, advection_v, self.ge, self.gi, dt, e_eq)
        self.compute_wb_source_2(0.5 * dt)

        self.fe = self.fe_eq + self.ge
        self.fi = self.fi_eq + self.gi


    def project(self, eq_manager, projection_type):
        mesh_x, mesh_v, fe, fi = (self.mesh_x, self.mesh_v,
            self.fe, self.fi)
        projection_manager = self.projection_manager

        if projection_type == "BGK":
            rho = compute_rho(mesh_v, fi - fe)
            phi = compute_phi(mesh_x, -rho)
            self.e_projection = compute_e(mesh_x, rho)
            fe_project, coef_fe = projection_manager.project(fe, -phi)
            fi_project, coef_fi = projection_manager.project(fi, phi)

            int_fe_project = integrate(mesh_x, mesh_v, fe_project)
            int_fi_project = integrate(mesh_x, mesh_v, fi_project)
            int_mean = (int_fe_project + int_fi_project) / 2.
            scale_coef_fe = int_mean / int_fe_project
            scale_coef_fi = int_mean / int_fi_project
            fe_project *= scale_coef_fe
            fi_project *= scale_coef_fi
            rho_project = compute_rho(mesh_v, fi_project - fe_project)
            e_project = compute_e(mesh_x, rho_project)

            self.fe_eq = fe_project
            self.fi_eq = fi_project
            self.dx_fe_eq, self.dv_fe_eq = projection_manager.get_df(coef_fe, -phi, -self.e_projection)
            self.dx_fi_eq, self.dv_fi_eq = projection_manager.get_df(coef_fi, phi, self.e_projection)
            self.dx_fe_eq *= scale_coef_fe
            self.dx_fi_eq *= scale_coef_fi
            self.dv_fe_eq *= scale_coef_fe
            self.dv_fi_eq *= scale_coef_fi
            self.e_eq = e_project
            self.ge = fe - self.fe_eq
            self.gi = fi - self.fi_eq
        elif projection_type == "coefficients":
            (self.fe_eq, self.fi_eq, self.dx_fe_eq, self.dx_fi_eq,
               self.dv_fe_eq, self.dv_fi_eq, self.e_eq) = self.optimizer.optimize_coef(self.fe, self.fi, eq_manager)
            self.ge = fe - self.fe_eq
            self.gi = fi - self.fi_eq
        else:
            raise ValueError(f"Projection type {projection_type} does not exist.")


    def compute_iteration(self, dt):
        if not self.wb_scheme:
            advect_vlasov(self.advection_x, self.advection_v,
                self.fe, self.fi, dt, order=2)
        else:
            #self.compute_wb_vlasov(dt)
            self.compute_wb_vlasov_2(dt)

