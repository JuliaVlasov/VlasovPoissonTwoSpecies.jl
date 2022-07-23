using VlasovPoissonTwoSpecies
import SpecialFunctions: ellipk

const refine_dt              = 1
const refine_factor          = 1
const eps_perturbation       = 0.001


data                   = Data()
data.wb_scheme         = true
data.projection        = false
data.projection_type   = :coefficients   # :BGK

coef = Coef()

#coef = Coef(solution = :JacobiDN, lambda = 1., a = -1., b = 1 + sqrt(2), c = 1., m = 0.97, x0 = 0.)
#coef = Coef(solution = :JacobiDN, lambda = 1., a = -2., b = 1 + sqrt(2), c = 1., m = 0.97, x0 = 0 )
#coef = Coef(solution = :JacobiND, lambda = 1., a = -1., b = 2 * (1 + sqrt(2)), c = 1., m = 0.75, x0= 0.)
#coef = Coef(solution = :JacobiND, lambda = 1., a = -2., b = 2 * (1 + sqrt(2)), c = 1., m = 0.75, x0 = 0.)
#coef = Coef(solution = :JacobiND, lambda = 1., a = -1., b = 2 * (1 + sqrt(2)), c = 1., m = 0.75,x0 = 0.)
#coef = Coef(solution = :JacobiCN, lambda = 1., a = -1., b = 3, c = 1., m = 0.97, x0 = 0)

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
 
vlasov_poisson(data, mesh_x, mesh_v, coef)
