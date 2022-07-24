using VlasovPoissonTwoSpecies
using Test
using FFTW

@testset "Poisson solver" begin

    nx, nv = 128, 256
    xmin, xmax = 0.0, 4π
    vmin, vmax = -6., 6.
    meshx = Mesh(xmin, xmax, nx)
    meshv = Mesh(vmin, vmax, nv)
    x, v = meshx.x, meshv.x
      
    # Set distribution function for Landau damping
    ϵ, kx = 0.001, 0.5
    f = landau( ϵ, kx, meshx, meshv)
    ρ = compute_rho(meshv, f)
    @show size(ρ)
    e = compute_e(meshx, ρ)
    @show size(e)
    
    @test ρ ≈ ϵ .* cos.(kx .* meshx.x)
    @test e ≈ ϵ .* sin.(kx .* meshx.x) ./ kx

end
