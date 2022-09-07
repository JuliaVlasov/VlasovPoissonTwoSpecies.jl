```@meta
CurrentModule = VlasovPoissonTwoSpecies
```

# Numerical scheme

This type contains data for the numerical scheme 

```math
f^{\pm}(t,x,v) = f_0^{\pm}(x,v) + g^{\pm}(t,x,v),
```

where ``f_0^{\pm}(x,v)`` a stationary solution 

```math
T_{\phi} = v \partial_x - \partial_x \phi \partial_v
```
this term is computed in [`T_f`](@ref)

```math
\begin{aligned}
   & \partial_t g^+ +  T_{\phi_f} g^+  = \partial_x \phi_g \partial_v f_0^+,\\
   &     \partial_t g^{-} +  \mu T_{- \phi_f} g^- = -\partial_x \phi_g \partial_v f_0^-,
  \end{aligned}
```

where ``\phi_g := \phi_f- \phi_{f_0}`` is the potential associated with ``g``. The two potentials satisfy the following equations

```math
    \partial_{xx} \phi_f = \int_{\mathbb{R}}(f^- - f^+) dv, \quad \partial_{xx} \phi_g = \int_{\mathbb{R}}(g^- - g^+) dv,
```
and the initial condition for ``g^{\pm}`` is given by
```math
  \begin{aligned}
    g^+(0,x,v) = f_{in}^+(x,v) - f_0^+(x,v), \quad     g^-(0,x,v) = f_{in}^-(x,v) - f_0^-(x,v).
  \end{aligned}
```
We use a time splitting scheme where the term ``\pm\partial_x \phi_g \partial_v f_0^\pm`` is treated as a source term with three main steps (with ``\mu=1``)

```math
  \begin{aligned}
    &(1) \quad \partial_t g^{\pm} = \pm \partial_x \phi_g \partial_v f_0^\pm,\\
    &(2) \quad \partial_t g^{\pm} + v \partial_x g^{\pm} = 0,\\
    &(3) \quad \partial_t g^{\pm} \mp \partial_x \phi_f\partial_v g^{\pm} = 0.
  \end{aligned}
```


We consider a second order in time discretization to solve (2) and (3) from ``t^n`` to ``t^{n+1}``. In the following ``\Delta t`` denotes the time step, ``g^{n}`` the semi-discretization in time of the unknown ``g`` at ``t^n := n \Delta t`` and ``g^{(k)}`` intermediate steps between ``t^n`` and ``t^{n+1}``. For simplicity we drop the index ``\pm`` but it is important to note that the steps of the procedure described below are computed at the same time for ``g^+`` and ``g^-``. This is require to solve the Poisson equation when evaluating ``\partial_x \phi_g``. One obtains the following algorithm

- Solve step (1) over a half time step with [`compute_source`](@ref): ``g^{(1)} = g^n + \frac{\Delta t}{2} \partial_x \phi_{g^n} \partial_v f_0`` where ``\partial_x\phi_{g^n}`` is calculated with Poisson equation,
- Solve step (2) over a half time step: ``g^{(2)}(x,v) = g^{(1)}(x-v\frac{\Delta t}{2}, v)``, with [`advect`](@ref).
- Update the value of ``E^{(2)} := -\partial_x \phi_{f^{(2)}}`` with Poisson solver and ``f^{(2)} := f_0 + g^{(2)}``, with [`compute_rho`](@ref) and [`compute_e`](@ref).
- Solve step (3) over a whole time step: ``g^{(3)}(x,v) = g^{(2)}(x,v-E^{(2)} \Delta t)``
- Solve step (2) over a half time step: ``g^{(4)}(x,v) = g^{3}(x-v\frac{\Delta t}{2},v)``
- Solve step (1) over a half time step: ``g^{(5)} = g^{(4)} + \frac{\Delta t}{2} \partial_x \phi_{g^{(4)}} \partial_v f_0`` where ``\partial_x\phi_{g^{(4)}}`` is calculated with Poisson solver.


```julia

# step 1
compute_source(scheme, 0.5dt)

# step 2
advect(scheme.advection_x, scheme.ge, v, 0.5dt)
advect(scheme.advection_x, scheme.gi, v, 0.5dt)

rho .= compute_rho(scheme)
e .= compute_e(mesh_x, rho)
e .+= scheme.e_eq

# step 3
advect(scheme.advection_v, transpose(scheme.ge), -e, dt)
advect(scheme.advection_v, transpose(scheme.gi), e, dt)

# step 2
advect(scheme.advection_x, scheme.ge, v, 0.5dt)
advect(scheme.advection_x, scheme.gi, v, 0.5dt)

scheme.fe .= scheme.fe_eq .+ scheme.ge
scheme.fi .= scheme.fi_eq .+ scheme.gi

# step 1
compute_source(scheme, 0.5dt)
```
