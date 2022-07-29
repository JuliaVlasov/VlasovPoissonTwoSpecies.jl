var documenterSearchIndex = {"docs":
[{"location":"api/#Documentation","page":"API","title":"Documentation","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"","category":"page"},{"location":"api/","page":"API","title":"API","text":"Modules = [VlasovPoissonTwoSpecies]","category":"page"},{"location":"api/#VlasovPoissonTwoSpecies.Advection","page":"API","title":"VlasovPoissonTwoSpecies.Advection","text":"struct Advection\n\nAdvection to be computed on each row\n\nmesh::Mesh\np::Int64\nmodes::Vector{Float64}\neig_bspl::Vector{Float64}\neigalpha::Vector{ComplexF64}\n\n\n\n\n\n","category":"type"},{"location":"api/#VlasovPoissonTwoSpecies.Coef","page":"API","title":"VlasovPoissonTwoSpecies.Coef","text":"mutable struct Coef\n\nsolution::Symbol\nlambda::Float64\na::Float64\nb::Float64\nc::Float64\nm::Float64\nx0::Float64\nalpha::Float64\nbeta::Float64\ngamma::Float64\ndelta::Float64\nepsilon::Float64\n\n\n\n\n\n","category":"type"},{"location":"api/#VlasovPoissonTwoSpecies.EquilibriumManager","page":"API","title":"VlasovPoissonTwoSpecies.EquilibriumManager","text":"struct EquilibriumManager\n\nmesh_x::Mesh\nmesh_v::Mesh\ncoef::Coef\nfe::Matrix{Float64}\nfi::Matrix{Float64}\ndx_fe::Matrix{Float64}\ndv_fe::Matrix{Float64}\ndx_fi::Matrix{Float64}\ndv_fi::Matrix{Float64}\n\n\n\n\n\n","category":"type"},{"location":"api/#VlasovPoissonTwoSpecies.OutputManager","page":"API","title":"VlasovPoissonTwoSpecies.OutputManager","text":"struct OutputManager\n\ndata::Data\nmesh_x::Mesh\nmesh_v::Mesh\nnb_outputs::Int64\nt::Vector{Float64}\ncompute_energy_eq::Bool\nenergy_fe_init::Float64\nenergy_fi_init::Float64\nenergy_fe::Vector{Float64}\nenergy_fi::Vector{Float64}\n\nData structure to manage outputs.\n\n\n\n\n\n","category":"type"},{"location":"api/#VlasovPoissonTwoSpecies.Scheme","page":"API","title":"VlasovPoissonTwoSpecies.Scheme","text":"struct Scheme\n\nmesh_x::Mesh\nmesh_v::Mesh\nadvection_x::VlasovPoissonTwoSpecies.Advection\nadvection_v::VlasovPoissonTwoSpecies.Advection\nfe::Matrix{Float64}\nfi::Matrix{Float64}\nge::Matrix{Float64}\ngi::Matrix{Float64}\nρ_eq::Vector{Float64}\ne_eq::Vector{Float64}\nfe_eq::Matrix{Float64}\nfi_eq::Matrix{Float64}\ndx_fe_eq::Matrix{Float64}\ndx_fi_eq::Matrix{Float64}\ndv_fe_eq::Matrix{Float64}\ndv_fi_eq::Matrix{Float64}\nwb_scheme::Bool\ne_projection::Vector{Float64}\n\n\n\n\n\n","category":"type"},{"location":"api/#VlasovPoissonTwoSpecies.T_f","page":"API","title":"VlasovPoissonTwoSpecies.T_f","text":"T_f(mesh_x, mesh_v, f, e, dx_f, dv_f)\nT_f(mesh_x, mesh_v, f, e, dx_f, dv_f, order)\n\n\n(vdx+edv) f\n\n\n\n\n\n","category":"function"},{"location":"api/#VlasovPoissonTwoSpecies.advect-NTuple{4, Any}","page":"API","title":"VlasovPoissonTwoSpecies.advect","text":"advect(self, f, v, dt)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.advect_vlasov","page":"API","title":"VlasovPoissonTwoSpecies.advect_vlasov","text":"advect_vlasov(advection_x, advection_v, fe, fi, dt)\nadvect_vlasov(advection_x, advection_v, fe, fi, dt, e_eq)\nadvect_vlasov(\n    advection_x,\n    advection_v,\n    fe,\n    fi,\n    dt,\n    e_eq,\n    order\n)\n\n\n\n\n\n\n","category":"function"},{"location":"api/#VlasovPoissonTwoSpecies.bspline-Tuple{Int64, Int64, Float64}","page":"API","title":"VlasovPoissonTwoSpecies.bspline","text":"bspline(p, j, x)\n\n\nReturn the value at x in [0,1] of the B-spline with integer nodes of degree p with support starting at j.  Implemented recursively using the  De Boor's Algorithm\n\nB_i0(x) = left\nbeginmatrix\n1  mathrmif  quad t_i  x  t_i+1 \n0  mathrmotherwise \nendmatrix\nright\n\nB_ip(x) = fracx - t_it_i+p - t_i B_ip-1(x) \n+ fract_i+p+1 - xt_i+p+1 - t_i+1 B_i+1p-1(x)\n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.compute_e-Tuple{Mesh, Vector{Float64}}","page":"API","title":"VlasovPoissonTwoSpecies.compute_e","text":"compute_e(mesh_x, rho)\n\n\ncompute Ex using that \n\n-ikE_x = rho \n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.compute_energy-Tuple{Any, Any, Any}","page":"API","title":"VlasovPoissonTwoSpecies.compute_energy","text":"compute_energy(mesh_x, mesh_v, f)\n\n\ne_f = int v^2 f dv\n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.compute_normalized_energy-Tuple{Any, Any, Any}","page":"API","title":"VlasovPoissonTwoSpecies.compute_normalized_energy","text":"compute_normalized_energy(self, fe, fi)\n\n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.compute_rho-Tuple{Mesh, Matrix{Float64}}","page":"API","title":"VlasovPoissonTwoSpecies.compute_rho","text":"compute_rho(meshv, f)\n\n\nCompute charge density\n\nrho(xt) = int f(xvt) dv\n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.get_normalized_energy-NTuple{4, Any}","page":"API","title":"VlasovPoissonTwoSpecies.get_normalized_energy","text":"get_normalized_energy(mesh_x, mesh_v, f, energy_eq)\n\n\nreturns the normalized energy \n\ne_f - e_eq  e_eq\n\nwhere e_f is the energy of f.\n\n\n\n\n\n","category":"method"},{"location":"api/#VlasovPoissonTwoSpecies.landau-NTuple{4, Any}","page":"API","title":"VlasovPoissonTwoSpecies.landau","text":"landau(ϵ, k, meshx, meshv)\n\n\nLandau Damping\n\nf(xv) = frac1sqrt2pi(1 + epsilon cos ( k x )) exp (- fracv^22 )\n\nLandau damping - Wikipedia\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = VlasovPoissonTwoSpecies","category":"page"},{"location":"#VlasovPoissonTwoSpecies","page":"Home","title":"VlasovPoissonTwoSpecies","text":"","category":"section"}]
}
