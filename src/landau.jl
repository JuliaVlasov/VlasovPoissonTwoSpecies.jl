export landau

"""
$(SIGNATURES)

# Landau Damping

```math
f(x,v) = \\frac{1}{\\sqrt{2\\pi}}(1 + \\epsilon \\cos ( k x )) \\exp (- \\frac{v^2}{2} )
```

[Landau damping - Wikipedia](https://en.wikipedia.org/wiki/Landau_damping)
"""
function landau(ϵ, k, meshx, meshv)
    nx = meshx.nx
    nv = meshv.nx
    x = meshx.x
    v = meshv.x
    f = zeros(Float64, (nx, nv))
    f .= (1.0 .+ ϵ * cos.(k * x)) / sqrt(2π) .* transpose(exp.(-0.5 * v .^ 2))
    f
end
