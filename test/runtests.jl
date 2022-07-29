using VlasovPoissonTwoSpecies
using Test
using FFTW
using PyCall
import SpecialFunctions: ellipk

@testset "Equilibrium" begin

pushfirst!(PyVector(pyimport("sys")."path"), "")

py"""
import sys, os
import numpy as np
from scipy.special import ellipk

from data import Data
from uniform_mesh import Mesh
from vlasov_poisson import get_equilibriums, perturbate_func
from equilibrium_manager import EquilibriumManager
from output_manager import OutputManager, compute_energy
from scheme import Scheme
from tool_box import compute_rho, compute_e

refine_dt              = 1
refine_factor          = 1
eps_perturbation       = 0.001

data                   = Data()
data.wb_scheme         = True
data.projection        = False
data.projection_type   = "coefficients"
data.coef              = {"solution type": "JacobiDN", "lambda": 1., "a": -1., "b": 1 + np.sqrt(2) + 0.12,
                          "c": 1., "m": 0.97, "x0": 0.}
data.T_final           = 200
data.nb_time_steps     = 1000 * refine_factor * refine_dt
data.nx                = 64 * refine_factor
data.nv                = 64 * refine_factor
data.x_min             = 0
data.x_max             = 4 * ellipk(data.coef["m"])
data.v_min             = -10
data.v_max             = 10
data.perturbation_init = lambda x, v: eps_perturbation * (
    np.cos(2 * np.pi * x / (data.x_max - data.x_min) + 1))
data.output            = True
data.vtk               = True
data.freq_save         = 5 * refine_factor * refine_dt
data.freq_output       = 5 * refine_factor * refine_dt
data.freq_projection   = 5 * 5 * refine_factor * refine_dt

mesh_x = Mesh(data.x_min, data.x_max, data.nx)
mesh_v = Mesh(data.v_min, data.v_max, data.nv)

tf = data.T_final
nt = data.nb_time_steps
dt = tf / nt

eq_manager = EquilibriumManager(mesh_x, mesh_v, data.coef)
output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe_eq, eq_manager.fi_eq)

fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq = get_equilibriums(
    eq_manager, perturbated=False, epsilon=1e-2)

fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe_eq,
    data.perturbation_init)
fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi_eq,
    data.perturbation_init)


scheme = Scheme(mesh_x, mesh_v, fe, fi, fe_eq, fi_eq,
dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq, data.wb_scheme)

rho = compute_rho(mesh_v, fi - fe)
e = compute_e(mesh_x, rho)

v = mesh_v.x

"""


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

    mesh_x = Mesh(data.x_min, data.x_max, data.nx)
    mesh_v = Mesh(data.v_min, data.v_max, data.nv)
     
    tf = data.T_final
    nt = data.nb_time_steps
    dt = tf / nt
    
    eq_manager = EquilibriumManager(coef, mesh_x, mesh_v)
    output = OutputManager(data, mesh_x, mesh_v, eq_manager.fe, eq_manager.fi)

    @test eq_manager.fe ≈ py"eq_manager.fe_eq"
    @test eq_manager.fi ≈ py"eq_manager.fi_eq"

    energy_fe_init = compute_energy(mesh_x, mesh_v, eq_manager.fe)
    energy_fi_init = compute_energy(mesh_x, mesh_v, eq_manager.fi)

    @test energy_fe_init ≈ py"compute_energy(mesh_x, mesh_v, eq_manager.fe_eq)"
    @test energy_fi_init ≈ py"compute_energy(mesh_x, mesh_v, eq_manager.fi_eq)"

    @test output.energy_fe_init ≈ py"output.energy_fe_eq_init"
    @test output.energy_fi_init ≈ py"output.energy_fi_eq_init"
    
    fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq =
        get_equilibriums(eq_manager, perturbated = false, epsilon = 1e-2)

    @test fe_eq ≈ py"fe_eq"
    @test fi_eq ≈ py"fi_eq"
    @test dx_fe_eq ≈ py"dx_fe_eq"
    @test dx_fi_eq ≈ py"dx_fi_eq"
    @test dv_fe_eq ≈ py"dv_fe_eq"
    @test dv_fi_eq ≈ py"dv_fi_eq"
    
    fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe, data.perturbation_init)
    fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi, data.perturbation_init)

    @test fe ≈ py"fe"
    @test fi ≈ py"fi"

    ρ = compute_rho(mesh_v, fi .- fe)
    e = compute_e(mesh_x, ρ)

    @test ρ ≈ py"rho"
    @test e ≈ py"e"

    scheme = Scheme( mesh_x, mesh_v, fe, fi, fe_eq, fi_eq, dx_fe_eq, dx_fi_eq,
        dv_fe_eq, dv_fi_eq, data.wb_scheme,)

    v = mesh_v.x

    @test fe ≈ py"fe"
    @test fi ≈ py"fi"

py"""
energy_fe = []
energy_fi = []
for i in range(100):
    scheme.advection_x.advect(np.transpose(fe), v, 0.5*dt)
    scheme.advection_x.advect(np.transpose(fi), v, 0.5*dt)
    rho = compute_rho(mesh_v, fi - fe)
    e = compute_e(mesh_x, rho)
    scheme.advection_v.advect(fe, -e, dt)
    scheme.advection_v.advect(fi, e, dt)
    scheme.advection_x.advect(np.transpose(fe), v, 0.5*dt)
    scheme.advection_x.advect(np.transpose(fi), v, 0.5*dt)
    e_fe, e_fi = output.compute_normalized_energy(fe, fi)
    energy_fe.append(e_fe)
    energy_fi.append(e_fi)

"""

    energy_fe = Float64[]
    energy_fi = Float64[]
    for i in 1:100
        advect(scheme.advection_x, fe, v, 0.5dt)
        advect(scheme.advection_x, fi, v, 0.5dt)
        ρ = compute_rho(mesh_v, fi .- fe)
        e = compute_e(mesh_x, ρ)
        advect(scheme.advection_v, transpose(fe), -e, dt)
        advect(scheme.advection_v, transpose(fi), e, dt)
        advect(scheme.advection_x, fe, v, 0.5dt)
        advect(scheme.advection_x, fi, v, 0.5dt)
        e_fe, e_fi = compute_normalized_energy(output, fe, fi)
        push!(energy_fe, e_fe)
        push!(energy_fi, e_fi)
    end

    @test ρ ≈ py"rho"
    @test e ≈ py"e"
    @test fe ≈ py"fe"
    @test fi ≈ py"fi"
    @test energy_fe ≈ py"energy_fe"
    @test energy_fi ≈ py"energy_fi"
    
end

@testset "Poisson solver" begin

    nx, nv = 128, 256
    xmin, xmax = 0.0, 4π
    vmin, vmax = -6.0, 6.0
    meshx = Mesh(xmin, xmax, nx)
    meshv = Mesh(vmin, vmax, nv)
    x, v = meshx.x, meshv.x

    # Set distribution function for Landau damping
    ϵ, kx = 0.001, 0.5
    f = landau(ϵ, kx, meshx, meshv)
    ρ = compute_rho(meshv, f)
    e = compute_e(meshx, ρ)

    @test ρ ≈ ϵ .* cos.(kx .* meshx.x)
    @test e ≈ ϵ .* sin.(kx .* meshx.x) ./ kx

end

