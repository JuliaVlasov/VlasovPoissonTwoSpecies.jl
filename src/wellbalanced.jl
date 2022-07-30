export WellBalanced

"""
$(TYPEDEF)

$(TYPEDFIELDS)
"""
struct WellBalanced <: AbstractScheme

    mesh_x::Mesh
    mesh_v::Mesh
    advection_x::Advection
    advection_v::Advection
    fe::Matrix{Float64}
    fi::Matrix{Float64}
    ge::Matrix{Float64}
    gi::Matrix{Float64}
    geᵗ::Matrix{Float64}
    giᵗ::Matrix{Float64}
    ρ_eq::Vector{Float64}
    e_eq::Vector{Float64}
    fe_eq::Matrix{Float64}
    fi_eq::Matrix{Float64}
    dx_fe_eq::Matrix{Float64}
    dx_fi_eq::Matrix{Float64}
    dv_fe_eq::Matrix{Float64}
    dv_fi_eq::Matrix{Float64}
    t_f::Matrix{Float64}

    function WellBalanced(fe, fi, equilibrium)

        advection_x = Advection(equilibrium.mesh_x)
        advection_v = Advection(equilibrium.mesh_v)

        ρ_eq = zeros(equilibrium.mesh_x.nx)
        e_eq = zeros(equilibrium.mesh_x.nx)
        ge = zero(equilibrium.fe)
        gi = zero(equilibrium.fi)

        ρ_eq .= compute_rho(equilibrium.mesh_v, equilibrium.fi .- equilibrium.fe)
        e_eq .= compute_e(equilibrium.mesh_x, ρ_eq)
        ge .= fe .- equilibrium.fe
        gi .= fi .- equilibrium.fi

        geᵗ = zeros(eltype(ge), reverse(size(ge)))
        giᵗ = zeros(eltype(gi), reverse(size(gi)))

        t_f = zeros(equilibrium.mesh_x.nx, equilibrium.mesh_v.nx)

        new(
            equilibrium.mesh_x,
            equilibrium.mesh_v,
            advection_x,
            advection_v,
            fe,
            fi,
            ge,
            gi,
            geᵗ,
            giᵗ,
            ρ_eq,
            e_eq,
            equilibrium.fe,
            equilibrium.fi,
            equilibrium.dx_fe,
            equilibrium.dx_fi,
            equilibrium.dv_fe,
            equilibrium.dv_fi,
            t_f
        )

    end

end

function integrate(mesh_x, mesh_v, f)

    dx = mesh_x.dx
    dv = mesh_v.dx
    dv * dx * sum(f)

end



export perturbate!

function perturbate!(scheme :: WellBalanced, func; ϵ = 1e-3)

    mesh_x = scheme.mesh_x
    mesh_v = scheme.mesh_v
    x_min = scheme.mesh_x.x_min
    x_max = scheme.mesh_x.x_max

    L = 2π / (x_max - x_min)
    p = (xi, vj) -> ϵ * (cos(L * xi + 1.0) + 0.5 * sin(2 * L * xi - 0.5))
    dx_p =
        (xi, vj) ->
            ϵ * (-L * sin(L * xi + 1.0) + 0.5 * 2 * L * cos(2 * L * xi - 0.5)) - 1

    scale_coef =
        integrate(mesh_x, mesh_v, scheme.fi_eq) / integrate(mesh_x, mesh_v, scheme.fe_eq)


    scheme.dv_fe_eq .= func(mesh_x, mesh_v, scheme.dv_fe_eq, p)
    scheme.dv_fi_eq .= func(mesh_x, mesh_v, scheme.dv_fi_eq, p)

    scheme.dx_fe_eq .= (
        func(mesh_x, mesh_v, scheme.dx_fe_eq, p) .+
        func(mesh_x, mesh_v, scheme.fe_eq, dx_p)
    )

    scheme.dx_fi_eq .= (
        func(mesh_x, mesh_v, scheme.dx_fi_eq, p) .+
        func(mesh_x, mesh_v, scheme.fi_eq, dx_p)
    )

    scheme.fe_eq .= func(mesh_x, mesh_v, scheme.fe_eq, p)
    scheme.fi_eq .= func(mesh_x, mesh_v, scheme.fi_eq, p)

    scheme.fe_eq .*= scale_coef
    scheme.dv_fe_eq .*= scale_coef
    scheme.dx_fe_eq .*= scale_coef

end

export compute_source

"""
$(SIGNATURES)

compute 
```math
(vdx+edv) f
```

"""
function compute_source(scheme :: WellBalanced, dt)

    rho = compute_rho(scheme.mesh_v, (scheme.fi_eq .+ scheme.gi) .- (scheme.fe_eq .+ scheme.ge))
    e = compute_e(scheme.mesh_x, rho)
    v = scheme.mesh_v.x

    scheme.t_f .= scheme.dx_fe_eq .* v'
    scheme.t_f .-= scheme.dv_fe_eq .* e

    scheme.ge .-= dt .* scheme.t_f

    scheme.t_f .= scheme.dx_fi_eq .* v'
    scheme.t_f .+= scheme.dv_fi_eq .* e

    scheme.gi .-= dt .* scheme.t_f

end

export advect

compute_rho( scheme :: WellBalanced ) = compute_rho(scheme.mesh_v, scheme.gi .- scheme.ge)
