using VlasovPoissonTwoSpecies
import SpecialFunctions: ellipk

const refine_dt              = 1
const refine_factor          = 1
const eps_perturbation       = 0.001


data                   = Data()
data.wb_scheme         = true
data.projection        = false
data.projection_type   = :coefficients   # :BGK

coef = Coef()

#coef = Coef(solution = :JacobiDN, lambda = 1., a = -1., b = 1 + sqrt(2), c = 1., m = 0.97, x0 = 0.)
#coef = Coef(solution = :JacobiDN, lambda = 1., a = -2., b = 1 + sqrt(2), c = 1., m = 0.97, x0 = 0 )
#coef = Coef(solution = :JacobiND, lambda = 1., a = -1., b = 2 * (1 + sqrt(2)), c = 1., m = 0.75, x0= 0.)
#coef = Coef(solution = :JacobiND, lambda = 1., a = -2., b = 2 * (1 + sqrt(2)), c = 1., m = 0.75, x0 = 0.)
#coef = Coef(solution = :JacobiND, lambda = 1., a = -1., b = 2 * (1 + sqrt(2)), c = 1., m = 0.75,x0 = 0.)
#coef = Coef(solution = :JacobiCN, lambda = 1., a = -1., b = 3, c = 1., m = 0.97, x0 = 0)

data.T_final           = 200
data.nb_time_steps     = 1000 * refine_factor * refine_dt
data.nx                = 64 * refine_factor
data.nv                = 64 * refine_factor
data.x_min             = 0
data.x_max             = 4 * ellipk(coef.m)
data.v_min             = -10
data.v_max             = 10
data.perturbation_init = (x, v) -> eps_perturbation * (cos(2π * x / (data.x_max - data.x_min) + 1))
data.output            = true
data.vtk               = true
data.freq_save         = 5 * refine_factor * refine_dt
data.freq_output       = 5 * refine_factor * refine_dt
data.freq_projection   = 5 * 5 * refine_factor * refine_dt

mesh_x = Mesh(data.x_min, data.x_max, data.nx)
mesh_v = Mesh(data.v_min, data.v_max, data.nv)
 
tf = data.T_final
nt = data.nb_time_steps
dt = tf / nt

eq_manager = EquilibriumManager(coef, mesh_x, mesh_v)
#output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe_eq, eq_manager.fi_eq)

fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq =
    get_equilibriums(eq_manager, perturbated = false, epsilon = 1e-2)

fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe, data.perturbation_init)
fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi, data.perturbation_init)

scheme = Scheme(
    mesh_x,
    mesh_v,
    fe,
    fi,
    fe_eq,
    fi_eq,
    dx_fe_eq,
    dx_fi_eq,
    dv_fe_eq,
    dv_fi_eq,
    data.wb_scheme,
)

for it = 1:nt

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
            save(output,
                scheme.fe,
                scheme.fi,
                scheme.fe_eq,
                scheme.fi_eq,
                it * dt,
            )
        end
    end

    compute_iteration(scheme, dt)

end

end_of_simulation(output)

if data.output
    println("-"^40)
    println(" Computation ends at T = $(data.T_final) ")
    println("-"^40)
end
