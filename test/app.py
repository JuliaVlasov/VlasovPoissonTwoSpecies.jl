import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk
from data import Data
from uniform_mesh import Mesh
from vlasov_poisson import get_equilibriums, perturbate_func
from equilibrium_manager import EquilibriumManager
from scheme import Scheme
from output_manager import OutputManager

refine_dt              = 1
refine_factor          = 1
eps_perturbation       = 0.001

data                   = Data()
data.wb_scheme         = True
data.projection        = False
data.projection_type   = "coefficients"#"BGK"#
data.coef              = {"solution type": "JacobiDN", "lambda": 1., "a": -1., "b": 1 + np.sqrt(2) + 0.12,
                          "c": 1., "m": 0.97, "x0": 0.}

data.T_final           = 200
data.nb_time_steps     = 1000 * refine_factor * refine_dt
data.nx                = 64 * refine_factor
data.nv                = 64 * refine_factor
data.x_min             = 0
data.x_max             = 4 * ellipk(data.coef["m"])
data.v_min             = -10
data.v_max             = 10
data.perturbation_init = lambda x, v: eps_perturbation * (
    np.cos(2 * np.pi * x / (data.x_max - data.x_min) + 1))
data.output            = True
data.vtk               = True
data.freq_save         = 5 * refine_factor * refine_dt
data.freq_output       = 5 * refine_factor * refine_dt
data.freq_projection   = 5 * 5 * refine_factor * refine_dt

mesh_x = Mesh(data.x_min, data.x_max, data.nx)
mesh_v = Mesh(data.v_min, data.v_max, data.nv)

tf = data.T_final
nt = data.nb_time_steps
dt = tf / nt

eq_manager = EquilibriumManager(mesh_x, mesh_v, data.coef)
output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe_eq, eq_manager.fi_eq)

fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq = get_equilibriums(
    eq_manager, perturbated=False, epsilon=1e-2)

fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe_eq,
    data.perturbation_init)
fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi_eq,
    data.perturbation_init)

scheme = Scheme(mesh_x, mesh_v, fe, fi, fe_eq, fi_eq,
dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq, data.wb_scheme)

for it in range(0, nt):

    print("Computing time ", it * dt)
    output.save_output(scheme.fe, scheme.fi, scheme.fe_eq, scheme.fi_eq, it * dt)

    scheme.compute_iteration(dt)

plt.plot(output.t, output.energy_fe)
