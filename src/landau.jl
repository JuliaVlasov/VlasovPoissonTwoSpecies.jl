export landau

"""
$(SIGNATURES)

# Landau Damping

[Landau damping - Wikipedia](https://en.wikipedia.org/wiki/Landau_damping)
"""
function landau(ϵ, kx, meshx, meshv)
    nx = meshx.nx
    nv = meshv.nx
    x = meshx.x
    v = meshv.x
    f = zeros(Float64, (nx, nv))
    f .= (1.0 .+ ϵ * cos.(kx * x)) / sqrt(2π) .* transpose(exp.(-0.5 * v .^ 2))
    f
end
