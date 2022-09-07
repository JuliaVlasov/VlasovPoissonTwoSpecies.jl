using FFTW
import Statistics: mean

abstract type AbstractScheme end

"""
$(SIGNATURES)
"""
function advect_vlasov(advection_x, advection_v, fe, fi, dt, e_eq = nothing, order = 2)

    mesh_x = advection_x.mesh
    mesh_v = advection_v.mesh

    v = mesh_v.x

    if order == 1

        rho = compute_rho(mesh_v, fi - fe)
        e = compute_e(mesh_x, rho)

        # In case of wb scheme we add e_eq to e.
        if !isnothing(e_eq)
            e .+= e_eq
        end

        advect(advection_v, transpose(fe), -e, dt)
        advect(advection_v, transpose(fi), e, dt)

        advect(advection_x, fe, v, dt)
        advect(advection_x, fi, v, dt)

    elseif order == 2

        advect(advection_x, fe, v, 0.5dt)
        advect(advection_x, fi, v, 0.5dt)

        rho = compute_rho(mesh_v, fi .- fe)
        e = compute_e(mesh_x, rho)

        # In case of wb scheme we add e_eq to e.
        if !isnothing(e_eq)
            e .+= e_eq
        end

        advect(advection_v, transpose(fe), -e, dt)
        advect(advection_v, transpose(fi), e, dt)

        advect(advection_x, fe, v, 0.5dt)
        advect(advection_x, fi, v, 0.5dt)

    end
end

export Scheme


"""
$(TYPEDEF)

$(TYPEDFIELDS)
"""
struct Scheme

    mesh_x::Mesh
    mesh_v::Mesh
    advection_x::Advection
    advection_v::Advection
    fe::Matrix{Float64}
    fi::Matrix{Float64}
    ge::Matrix{Float64}
    gi::Matrix{Float64}
    ρ_eq::Vector{Float64}
    e_eq::Vector{Float64}
    fe_eq::Matrix{Float64}
    fi_eq::Matrix{Float64}
    dx_fe_eq::Matrix{Float64}
    dx_fi_eq::Matrix{Float64}
    dv_fe_eq::Matrix{Float64}
    dv_fi_eq::Matrix{Float64}
    wb_scheme::Bool
    e_projection::Vector{Float64}

    function Scheme(
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
        wb_scheme,
    )

        advection_x = Advection(mesh_x)
        advection_v = Advection(mesh_v)

        ρ_eq = zeros(mesh_x.nx)
        e_eq = zeros(mesh_x.nx)
        ge = zero(fe)
        gi = zero(fi)

        if wb_scheme
            ρ_eq .= compute_rho(mesh_v, fi_eq .- fe_eq)
            e_eq .= compute_e(mesh_x, ρ_eq)
            ge .= fe .- fe_eq
            gi .= fi .- fi_eq
        end

        e_projection = similar(e_eq)

        new(
            mesh_x,
            mesh_v,
            advection_x,
            advection_v,
            fe,
            fi,
            ge,
            gi,
            ρ_eq,
            e_eq,
            fe_eq,
            fi_eq,
            dx_fe_eq,
            dx_fi_eq,
            dv_fe_eq,
            dv_fi_eq,
            wb_scheme,
            e_projection,
        )

    end

end


export compute_rho

"""
$(SIGNATURES)

Compute charge density

```math
\\rho(x,t) = \\int f(x,v,t) dv
```
"""
function compute_rho(meshv::Mesh, f::Array{Float64,2})

    dv = meshv.dx
    rho = dv * sum(f, dims = 2)
    vec(rho .- mean(rho)) # vec squeezes the 2d array returned by sum function

end

export compute_e

"""
$(SIGNATURES)

compute Ex using that 

```math
-ikE_x = \\rho 
```
"""
function compute_e(mesh_x::Mesh, rho::Vector{Float64})

    nx = mesh_x.nx
    k = 2pi / (mesh_x.x_max - mesh_x.x_min)
    modes = collect(k .* fftfreq(nx, nx))
    modes[begin] = 1.0

    fft_rho = fft(rho)
    fft_rho[begin] = 0.0
    rhok = fft_rho ./ modes
    rhok .*= -1im

    return real(ifft(rhok))

end

function compute_wb_source(scheme, dt)
    mesh_x, mesh_v, ge, gi, dv_fe_eq, dv_fi_eq =
        (scheme.mesh_x, scheme.mesh_v, scheme.ge, scheme.gi, scheme.dv_fe_eq, scheme.dv_fi_eq)
    rho_g = compute_rho(mesh_v, gi - ge)
    e_g = compute_e(mesh_x, rho_g)
    for i in eachindex(e_g)
        scheme.ge[i, :] .+= dt .* e_g[i] .* view(dv_fe_eq, i, :)
        scheme.gi[i, :] .-= dt .* e_g[i] .* view(dv_fi_eq, i, :)
    end
end

function compute_wb_vlasov(scheme, dt)

    advection_x, advection_v, e_eq = (scheme.advection_x, scheme.advection_v, scheme.e_eq)

    compute_wb_source(scheme, 0.5 * dt)
    advect_vlasov(advection_x, advection_v, scheme.ge, scheme.gi, dt, e_eq, order = 2)
    compute_wb_source(scheme, 0.5 * dt)
    scheme.fe .= scheme.fe_eq .+ scheme.ge
    scheme.fi .= scheme.fi_eq .+ scheme.gi

end

function compute_wb_vlasov_2(scheme, dt)

    (advection_x, advection_v, e_eq) = (scheme.advection_x, scheme.advection_v, scheme.e_eq)

    compute_wb_source_2(scheme, 0.5 * dt)
    advect_vlasov(advection_x, advection_v, scheme.ge, scheme.gi, dt, e_eq)
    compute_wb_source_2(scheme, 0.5 * dt)

    scheme.fe .= scheme.fe_eq .+ scheme.ge
    scheme.fi .= scheme.fi_eq .+ scheme.gi

end

export compute_iteration

function compute_iteration(scheme, dt)
    if scheme.wb_scheme
        compute_wb_vlasov_2(scheme, dt)
    else
        advect_vlasov(scheme.advection_x, scheme.advection_v, scheme.fe, scheme.fi, dt, order = 2)
    end
end

function get_df_FD_matrix(size_f, order = 8)

    A = zeros(size_f, size_f)

    if order == 2
        coef_right = [0.5]
    elseif order == 4
        coef_right = [2 / 3, -1 / 12]
    elseif order == 6
        coef_right = [3 / 4, -3 / 20, 1 / 60]
    elseif order == 8
        coef_right = [4 / 5, -1 / 5, 4 / 105, -1 / 280]
    else
        @error("The order $order is not implemented for centered finite difference.")
    end

    right_pos = [i + 1 for i = 0:length(coef_right)-1]

    for i = 0:size_f-1
        current_right_pos = [i + pos for pos in right_pos]
        current_right_pos = [p < size_f ? p : p - size_f for p in current_right_pos]

        current_left_pos = [i - pos for pos in right_pos]
        current_left_pos = [p >= 0 ? p : size_f + p for p in current_left_pos]

        for j = 0:length(coef_right)-1
            A[i+1, current_right_pos[j+1]+1] += coef_right[j+1]
            A[i+1, current_left_pos[j+1]+1] += -coef_right[j+1]
        end
    end

    return A

end


"""
$(SIGNATURES)

```math
T_{\\phi} = v \\partial_x - \\partial_x \\phi \\partial_v
```

"""
function T_f(mesh_x, mesh_v, f, e, dx_f, dv_f, order = 8)

    dx = mesh_x.dx
    dv = mesh_v.dx
    x = mesh_x.x
    v = mesh_v.x

    A_dx = get_df_FD_matrix(mesh_x.nx, order)
    A_dv = get_df_FD_matrix(mesh_v.nx, order)

    v_dx_f = dx_f .* v'

    e_dv_f = dv_f .* e

    return v_dx_f .+ e_dv_f
end

export compute_wb_source_2

function compute_wb_source_2(scheme, dt)

    (mesh_x, mesh_v, ge, gi, fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq) = (
        scheme.mesh_x,
        scheme.mesh_v,
        scheme.ge,
        scheme.gi,
        scheme.fe_eq,
        scheme.fi_eq,
        scheme.dx_fe_eq,
        scheme.dx_fi_eq,
        scheme.dv_fe_eq,
        scheme.dv_fi_eq,
    )

    e_projection = scheme.e_projection

    rho = compute_rho(mesh_v, (fi_eq .+ gi) .- (fe_eq .+ ge))
    e = compute_e(mesh_x, rho)

    T_fe_eq = T_f(mesh_x, mesh_v, fe_eq, -e, dx_fe_eq, dv_fe_eq)
    T_fi_eq = T_f(mesh_x, mesh_v, fi_eq, e, dx_fi_eq, dv_fi_eq)
    scheme.ge .-= dt .* T_fe_eq
    scheme.gi .-= dt .* T_fi_eq

    #    for i in eachindex(e)
    #        scheme.ge[i, :] += dt * (e[i] - e_projection[i]) * dv_fe_eq[i, :]
    #        scheme.gi[i, :] -= dt * (e[i] - e_projection[i]) * dv_fi_eq[i, :]
    #    end
end


#=

    def project(scheme, eq_manager, projection_type):
        mesh_x, mesh_v, fe, fi = (scheme.mesh_x, scheme.mesh_v,
            scheme.fe, scheme.fi)
        projection_manager = scheme.projection_manager

        if projection_type == "BGK":
            rho = compute_rho(mesh_v, fi - fe)
            phi = compute_phi(mesh_x, -rho)
            scheme.e_projection = compute_e(mesh_x, rho)
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

            scheme.fe_eq = fe_project
            scheme.fi_eq = fi_project
            scheme.dx_fe_eq, scheme.dv_fe_eq = projection_manager.get_df(coef_fe, -phi, -scheme.e_projection)
            scheme.dx_fi_eq, scheme.dv_fi_eq = projection_manager.get_df(coef_fi, phi, scheme.e_projection)
            scheme.dx_fe_eq *= scale_coef_fe
            scheme.dx_fi_eq *= scale_coef_fi
            scheme.dv_fe_eq *= scale_coef_fe
            scheme.dv_fi_eq *= scale_coef_fi
            scheme.e_eq = e_project
            scheme.ge = fe - scheme.fe_eq
            scheme.gi = fi - scheme.fi_eq
        elif projection_type == "coefficients":
            (scheme.fe_eq, scheme.fi_eq, scheme.dx_fe_eq, scheme.dx_fi_eq,
               scheme.dv_fe_eq, scheme.dv_fi_eq, scheme.e_eq) = scheme.optimizer.optimize_coef(scheme.fe, scheme.fi, eq_manager)
            scheme.ge = fe - scheme.fe_eq
            scheme.gi = fi - scheme.fi_eq
        else:
            raise ValueError(f"Projection type {projection_type} does not exist.")


=#
