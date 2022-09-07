```@meta
CurrentModule = VlasovPoissonTwoSpecies
```

# Vlasov-Poisson system

The system considered is the two species Vlasov-Poisson system which is a simple model used to describe a collisionless plasma.

```math
  \begin{cases}
    \partial_t f^+ + v \partial_xf^+ - \partial_x \phi \partial_v f^+ &= 0,\\
    \partial_t f^- + v \partial_xf^- + \mu \partial_x \phi \partial_v f^- &= 0,\\
  \end{cases}
```
where the potential ``\phi := \phi(x)`` satisfy
```math
   \partial_{xx} \phi = \int_{\mathbb{R}}(f^- - f^+) dv,
```

where ``\mu > 0`` is the mass ratio, ``f^+ := f^+(t,x,v)`` and ``f^- := f^-(t,x,v)`` denote the distribution function for the ions and electrons respectively. The system is given with an initial condition

```math
\begin{equation*}
  \begin{cases}
    f^+(0,x,v) = f_{in}^+(x,v),\\
        f^-(0,x,v) = f_{in}^{-}(x,v),\\
  \end{cases}
\end{equation*}
```

on the domain ``x \in [0, L]``, ``v \in \mathbb{R}`` where we assume periodic boundary conditions in ``x`` and vanishing boundary conditions in ``v``.

The well-balanced numerical scheme is using a micro-macro type decomposition, where the macro part corresponds to the equilibrium and the micro part corresponds to the out of equilibrium part. 
The numerical solution will be written as ``f^\pm = f_0 ^\pm + g^\pm`` where ``f_0^\pm``  is a given equilibrium and ``g^\pm`` can be seen as a perturbation of ``f_0^\pm`` The unknown of the reformulated problem will be the perturbation ``g^\pm``

To compute the initial solution, you can use the [`Coef`](@ref) type: 

```@example tutorial
using Plots
using VlasovPoissonTwoSpecies

coef = Coef()

x_min, x_max, nx = 0., 1., 64
v_min, v_max, nv = -10., 10, 64

mesh_x = Mesh(x_min, x_max, nx)
mesh_v = Mesh(v_min, v_max, nv)

x = mesh_x.x
v = mesh_v.x

eq = EquilibriumManager(coef, mesh_x, mesh_v)

p = plot(layout=(2), xlabel="x", ylabel="v")
contourf!(p[1], x, v, eq.fe, title="fe")
contourf!(p[2], x, v, eq.fi, title="fi")

```


