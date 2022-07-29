import numpy as np
from tool_box import integrate
from equilibrium_manager import EquilibriumManager
from scheme import Scheme
from output_manager import OutputManager


def perturbate_func(mesh_x, mesh_v, f, p):
    x = mesh_x.x
    v = mesh_v.x
    nx = mesh_x.nx
    nv = mesh_v.nx
    return np.array([[(1 + p(x[i], v[j])) * f[i, j]
                     for j in range(nv)] for i in range(nx)])


def get_equilibriums(eq_manager, perturbated=False, epsilon=1e-3):
    mesh_x = eq_manager.mesh_x
    mesh_v = eq_manager.mesh_v
    x_min = mesh_x.x_min
    x_max = mesh_x.x_max

    if perturbated:
        L = 2 * np.pi / (x_max - x_min)
        p = lambda xi, vj: epsilon * (np.cos(L * xi + 1.) + 0.5 * np.sin(2 * L * xi - 0.5))
        dx_p = lambda xi, vj: epsilon * (-L * np.sin(L * xi + 1.) + 0.5 * 2 * L * np.cos(2 * L * xi - 0.5)) - 1

        fe_eq = perturbate_func(mesh_x, mesh_v, eq_manager.fe_eq, p)
        fi_eq = perturbate_func(mesh_x, mesh_v, eq_manager.fi_eq, p)
        dv_fe_eq = perturbate_func(mesh_x, mesh_v, eq_manager.dv_fe_eq, p)
        dv_fi_eq = perturbate_func(mesh_x, mesh_v, eq_manager.dv_fi_eq, p)
        dx_fe_eq = (perturbate_func(mesh_x, mesh_v, eq_manager.dx_fe_eq, p)
            + perturbate_func(mesh_x, mesh_v, eq_manager.fe_eq, dx_p))
        dx_fi_eq = (perturbate_func(mesh_x, mesh_v, eq_manager.dx_fi_eq, p)
            + perturbate_func(mesh_x, mesh_v, eq_manager.fi_eq, dx_p))

        scale_coef = integrate(mesh_x, mesh_v, fi_eq) / integrate(mesh_x, mesh_v, fe_eq)
        fe_eq *= scale_coef
        dv_fe_eq *= scale_coef
        dx_fe_eq *= scale_coef
    else:
        fe_eq = eq_manager.fe_eq
        fi_eq = eq_manager.fi_eq
        dv_fe_eq = eq_manager.dv_fe_eq
        dv_fi_eq = eq_manager.dv_fi_eq
        dx_fe_eq = eq_manager.dx_fe_eq
        dx_fi_eq = eq_manager.dx_fi_eq

    return fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq


def vlasov_poisson(data, mesh_x, mesh_v):
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
        if data.output and it % data.freq_output == 0:
            print("Computing time ", it * dt)

        if data.wb_scheme and data.projection and (it % data.freq_projection == 0):
            if data.output and it % data.freq_output == 0 :
                print("Projecting...")

            scheme.project(eq_manager, data.projection_type)

        if it % data.freq_save == 0:
            if data.output:
                print("Saving solution...")
                output.save_output(scheme.fe, scheme.fi, scheme.fe_eq,
                    scheme.fi_eq, it * dt)

        scheme.compute_iteration(dt)

    output.end_of_simulation()

    if data.output:
        print("-" * 40)
        print(" Computation ends at T = ", data.T_final)
        print("-" * 40)
