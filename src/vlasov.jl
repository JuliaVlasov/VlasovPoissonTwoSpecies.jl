#from tool_box import integrate
#from equilibrium_manager import EquilibriumManager
#from scheme import Scheme
#from output_manager import OutputManager
#from profile_exec_time import profile


function perturbate_func(mesh_x, mesh_v, f, p)
    x = mesh_x.x
    v = mesh_v.x
    nx = mesh_x.nx
    nv = mesh_v.nx
    [(1 + p(x[i], v[j])) * f[i, j] for i in 1:nx, j in 1:nv]
end


function get_equilibriums(eq_manager; perturbated=false, epsilon=1e-3)

    mesh_x = eq_manager.mesh_x
    mesh_v = eq_manager.mesh_v
    x_min = mesh_x.x_min
    x_max = mesh_x.x_max

    if perturbated
        L = 2pi / (x_max - x_min)
        p = (xi, vj) -> epsilon * (cos(L * xi + 1.) + 0.5 * sin(2 * L * xi - 0.5))
        dx_p = (xi, vj) -> epsilon * (-L * sin(L * xi + 1.) + 0.5 * 2 * L * cos(2 * L * xi - 0.5)) - 1

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
    else
        fe_eq = eq_manager.fe_eq
        fi_eq = eq_manager.fi_eq
        dv_fe_eq = eq_manager.dv_fe_eq
        dv_fi_eq = eq_manager.dv_fi_eq
        dx_fe_eq = eq_manager.dx_fe_eq
        dx_fi_eq = eq_manager.dx_fi_eq
    end

    fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq
end

export vlasov_poisson

function vlasov_poisson(data, mesh_x, mesh_v, coef)

    tf = data.T_final
    nt = data.nb_time_steps
    dt = tf / nt

    eq_manager = EquilibriumManager(coef, mesh_x, mesh_v)
    #output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe_eq, eq_manager.fi_eq)

    fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq = get_equilibriums(
        eq_manager, perturbated=false, epsilon=1e-2)

    fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe_eq, data.perturbation_init)
    fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi_eq, data.perturbation_init)

    scheme = Scheme(mesh_x, mesh_v, fe, fi, fe_eq, fi_eq,
    dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq, data.wb_scheme)

    for it in 1:nt

        if data.output && it % data.freq_output == 0
            println("Computing time ", it * dt)
        end

        if data.wb_scheme && data.projection && (it % data.freq_projection == 0)
            if data.output && it % data.freq_output == 0
                println("Projecting...")
            end

            scheme.project(eq_manager, data.projection_type)
        end

        if it % data.freq_save == 0
            if data.output
                println("Saving solution...")
                output.save_output(scheme.fe, scheme.fi, scheme.fe_eq,
                    scheme.fi_eq, it * dt)
            end
        end

        scheme.compute_iteration(dt)

    end

    output.end_of_simulation()

    if data.output
        println("-" * 40)
        println(" Computation ends at T = ", data.T_final)
        println("-" * 40)
    end

end