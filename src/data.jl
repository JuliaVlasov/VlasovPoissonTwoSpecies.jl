export Coef

Base.@kwdef mutable struct Coef
	solution:: Symbol  = :JacobiDN
    lambda :: Float64 = 1.
    a :: Float64 =  -1.
    b :: Float64 = 1 + sqrt(2) + 0.12
    c :: Float64 = 1.
    m :: Float64 = 0.97
    x0 :: Float64 = 0.
end

getindex(c::Coef, s::String) = s

export Data

Base.@kwdef mutable struct Data

	T_final :: Int           = 1
	nb_time_steps :: Int     = 10
	nx :: Int                = 64
	nv :: Int                = 64
	x_min :: Float64             = 0
	x_max :: Float64             = 1
	v_min :: Float64             = -10
	v_max :: Int             = 10
	output :: Bool            = true
	vtk :: Bool               = true 
	freq_output :: Int       = 1
	freq_save :: Int         = 1
	wb_scheme :: Bool         = false
    perturbation_init :: Function = (x, v) -> 0.
	projection :: Bool        = false
	projection_type :: Symbol   = :coefficients
	freq_projection :: Int   = 1

end
