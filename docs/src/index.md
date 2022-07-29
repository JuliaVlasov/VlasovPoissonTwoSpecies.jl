```@meta
CurrentModule = VlasovPoissonTwoSpecies
```

# VlasovPoissonTwoSpecies

```@example vlasov
using Plots
using VlasovPoissonTwoSpecies
import SpecialFunctions: ellipk

const refine_dt              = 1
const refine_factor          = 1
const eps_perturbation       = 0.001

coef = Coef()

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
```

```@example vlasov
function run(coef, data)
    
     mesh_x = Mesh(data.x_min, data.x_max, data.nx)
     mesh_v = Mesh(data.v_min, data.v_max, data.nv)
      
     x = mesh_x.x
     v = mesh_v.x

     tf = data.T_final
     nt = data.nb_time_steps
     dt = tf / nt
     
     eq_manager = EquilibriumManager(coef, mesh_x, mesh_v)
     output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe, eq_manager.fi)
     
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

     rho = zeros(mesh_x.nx)
     e = zeros(mesh_x.nx)
     
     for it = 1:nt
     
         if it % data.freq_save == 0
             if data.output
                 save(output, scheme, it * dt)
             end
         end

         compute_wb_source_2(scheme, 0.5dt)

         advect(scheme.advection_x, scheme.ge, v, 0.5dt)
         advect(scheme.advection_x, scheme.gi, v, 0.5dt)

         rho .= compute_rho(mesh_v, scheme.gi .- scheme.ge)
         e .= compute_e(mesh_x, rho)

         e .+= scheme.e_eq

         advect(scheme.advection_v, transpose(scheme.ge), -e, dt)
         advect(scheme.advection_v, transpose(scheme.gi), e, dt)

         advect(scheme.advection_x, scheme.ge, v, 0.5dt)
         advect(scheme.advection_x, scheme.gi, v, 0.5dt)

         scheme.fe .= scheme.fe_eq .+ scheme.ge
         scheme.fi .= scheme.fi_eq .+ scheme.gi

         compute_wb_source_2(scheme, 0.5dt)

     end
     
     output

end
```

```@example vlasov
@time output = run(coef, data)
```

```@example vlasov
plot(output.t, output.energy_fe, label="electrons")
plot!(output.t, output.energy_fi, label="ions", legend=:topleft)
title!("Kinetic Energy")
```
