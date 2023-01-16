export OutputManager

"""
$(TYPEDEF)

$(TYPEDFIELDS)

Data structure to manage outputs.
"""
struct OutputManager

    data::Data
    mesh_x::Mesh
    mesh_v::Mesh
    nb_outputs::Int
    t::Vector{Float64}
    compute_energy_eq::Bool
    energy_fe_init::Float64
    energy_fi_init::Float64
    energy_fe::Vector{Float64}
    energy_fi::Vector{Float64}

    function OutputManager(data, mesh_x, mesh_v, fe_eq_init, fi_eq_init)

        nb_outputs = 0
        t = Float64[]

        compute_energy_eq = false

        energy_fe_init = compute_energy(mesh_x, mesh_v, fe_eq_init)
        energy_fi_init = compute_energy(mesh_x, mesh_v, fi_eq_init)

        energy_fe = Float64[]
        energy_fi = Float64[]

        new(
            data,
            mesh_x,
            mesh_v,
            nb_outputs,
            t,
            compute_energy_eq,
            energy_fe_init,
            energy_fi_init,
            energy_fe,
            energy_fi,
        )

    end

    OutputManager(data, equilibrium) = 
        OutputManager(data, equilibrium.mesh_x, equilibrium.mesh_v, 
                      equilibrium.fe, equilibrium.fi)

end

export compute_energy

"""
$(SIGNATURES)

```math
e_f = \\int v^2 f dv
```
"""
function compute_energy(mesh_x, mesh_v, f)

    v = mesh_v.x
    dx = mesh_x.dx
    dv = mesh_v.dx

    return dv * dx * sum(f .* v' .^ 2)

end


"""
$(SIGNATURES)

returns the normalized energy 

```math
|e_f - e_eq| / e_eq
```

where ``e_f`` is the energy of ``f``.
"""
function get_normalized_energy(mesh_x, mesh_v, f, energy_eq)
    return abs(compute_energy(mesh_x, mesh_v, f) - energy_eq) / energy_eq
end

export compute_normalized_energy

"""
$(SIGNATURES)
"""
function compute_normalized_energy(output, fe, fi)

    mesh_x = output.mesh_x
    mesh_v = output.mesh_v

    return (
        get_normalized_energy(mesh_x, mesh_v, fe, output.energy_fe_init),
        get_normalized_energy(mesh_x, mesh_v, fi, output.energy_fi_init),
    )

end

export save

function save(output, scheme, t)

    data = output.data
    mesh_x = output.mesh_x
    mesh_v = output.mesh_v

    push!(output.t, t)
    e_fe, e_fi = compute_normalized_energy(output, scheme.fe, scheme.fi)
    push!(output.energy_fe, e_fe)
    push!(output.energy_fi, e_fi)

end

export end_of_simulation

function end_of_simulation(output) end
