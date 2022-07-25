export Coef

"""
$(TYPEDEF)

$(TYPEDFIELDS)
"""
Base.@kwdef mutable struct Coef
    solution::Symbol = :JacobiDN
    lambda::Float64 = 1.0
    a::Float64 = -1.0
    b::Float64 = 1 + sqrt(2) + 0.12
    c::Float64 = 1.0
    m::Float64 = 0.97
    x0::Float64 = 0.0
    alpha::Float64 = 0.0
    beta::Float64 = 0.0
    gamma::Float64 = 0.0
    delta::Float64 = 0.0
    epsilon::Float64 = 0.0
end

function init!(coef::Coef)

    solution_type = coef.solution
    lambda_coef = coef.lambda
    c = coef.c
    m = coef.m

    if solution_type == :JacobiDN
        alpha = -lambda_coef^2 / (c^2)
        beta = 0.0
        gamma = (lambda_coef^2) * (2 - m)
        delta = 0.0
        epsilon = -(lambda_coef^2) * (1 - m) * c^2
    elseif solution_type == :JacobiND
        alpha = (lambda_coef^2) * (m - 1) / (c^2)
        beta = 0.0
        gamma = (lambda_coef^2) * (2 - m)
        delta = 0.0
        epsilon = -(lambda_coef^2) * c^2
    elseif solution_type == :JacobiCN
        alpha = -m * (lambda_coef^2) / (c^2)
        beta = 0.0
        gamma = (lambda_coef^2) * (2 * m - 1)
        delta = 0.0
        epsilon = (lambda_coef^2) * (1 - m) * c^2
    else
        @error("Solution type $solution_type not defined")
    end

    coef.alpha = alpha
    coef.beta = beta
    coef.gamma = gamma
    coef.delta = delta
    coef.epsilon = epsilon

end
