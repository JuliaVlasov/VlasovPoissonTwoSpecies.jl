using VlasovPoissonTwoSpecies
using Test
using FFTW
using Plots
import SpecialFunctions: ellipk
using ProgressMeter

refine_dt              = 1
refine_factor          = 1
eps_perturbation       = 0.001

coef                   = Coef()
data                   = Data()
data.wb_scheme         = true
data.projection        = false
data.projection_type   = :coefficients   # :BGK
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

function run(coef::Coef, data::Data)

    mesh_x = Mesh(data.x_min, data.x_max, data.nx)
    mesh_v = Mesh(data.v_min, data.v_max, data.nv)
     
    tf = data.T_final
    nt = data.nb_time_steps
    dt = tf / nt
    
    eq_manager = EquilibriumManager(coef, mesh_x, mesh_v)
    output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe, eq_manager.fi)
    
    fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq =
        get_equilibriums(eq_manager, perturbated = false, epsilon = 1e-2)
    
    fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe, data.perturbation_init)
    fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi, data.perturbation_init)
    
    scheme = Scheme( mesh_x, mesh_v, fe, fi, fe_eq, fi_eq, dx_fe_eq, dx_fi_eq,
        dv_fe_eq, dv_fi_eq, data.wb_scheme,)
    
    v = mesh_v.x
    
    e_fe, e_fi = compute_normalized_energy(output, fe, fi)
    
    @showprogress 1 for i in 1:nt
    
        compute_iteration(scheme, dt)
        
        save(output, scheme, i*dt)
    
    end

    output.t, output.energy_fe, output.energy_fi

end

time, energy_fe, energy_fi = run(coef, data)

plot(time, energy_fe)
plot!(time, energy_fi)

