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

fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq = get_equilibriums(
    eq_manager, perturbated=False, epsilon=1e-2)

fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe_eq,
    data.perturbation_init)
fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi_eq,
    data.perturbation_init)

"""

    fi_py = py"fi"
    fe_py = py"fe"

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
    
    fe_eq, fi_eq, dx_fe_eq, dx_fi_eq, dv_fe_eq, dv_fi_eq =
        get_equilibriums(eq_manager, perturbated = false, epsilon = 1e-2)
    
    fe = perturbate_func(mesh_x, mesh_v, eq_manager.fe, data.perturbation_init)
    fi = perturbate_func(mesh_x, mesh_v, eq_manager.fi, data.perturbation_init)

    @test fe ≈ fe_py
    @test fi ≈ fi_py

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
    @show size(ρ)
    e = compute_e(meshx, ρ)
    @show size(e)

    @test ρ ≈ ϵ .* cos.(kx .* meshx.x)
    @test e ≈ ϵ .* sin.(kx .* meshx.x) ./ kx

end

