import Elliptic: ellipj
using OMEinsum


function ellipj(u::Vector{Float64}, m::Float64)
    en = ellipj.(u, m)
    sn = getindex.(en, 1)
    cn = getindex.(en, 2)
    dn = getindex.(en, 3)
    sn, cn, dn
end

export EquilibriumManager

"""
$(TYPEDEF)

$(TYPEDFIELDS)
"""
struct EquilibriumManager

    mesh_x::Mesh
    mesh_v::Mesh
    coef::Coef
    fe::Matrix{Float64}
    fi::Matrix{Float64}
    dx_fe::Matrix{Float64}
    dv_fe::Matrix{Float64}
    dx_fi::Matrix{Float64}
    dv_fi::Matrix{Float64}

    function EquilibriumManager(coef, mesh_x, mesh_v, init_eq = true)

        init!(coef)

        if init_eq
            fe = fe_init(coef, mesh_x, mesh_v)
            fi = fi_init(coef, mesh_x, mesh_v)
            dx_fe, dv_fe = fe_prime_init(coef, mesh_x, mesh_v)
            dx_fi, dv_fi = fi_prime_init(coef, mesh_x, mesh_v)
        end

        new(mesh_x, mesh_v, coef, fe, fi, dx_fe, dv_fe, dx_fi, dv_fi)
    end
end

function get_exp_a_phi(coef, mesh_x)

    x = mesh_x.x
    lambda_coef = coef.lambda
    b = coef.b
    c = coef.c
    m = coef.m
    x0 = coef.x0
    sn, cn, dn = ellipj(lambda_coef .* x .+ x0, m)

    if coef.solution == :JacobiDN
        return (c .* dn .+ b)
    elseif coef.solution == :JacobiND
        return (c ./ dn) .+ b
    elseif coef.solution == :JacobiCN
        return (c .* cn .+ b)
    else
        @error("The solution $(coef.solution) does not exist.")
    end

end

function get_phi(coef, mesh_x)

    a = coef.a
    return log.(get_exp_a_phi(coef, mesh_x) .^ (1 / a))

end

function get_ode_coef(coef, mesh_x, scale_coef = false)

    a = coef.a
    b = coef.b

    alpha = coef.alpha
    beta = coef.beta
    gamma = coef.gamma
    delta = coef.delta
    epsilon = coef.epsilon

    r1 = -alpha / a
    r2 = -(-2 * alpha * b + (beta / 2)) / a
    s1 = (1 / a) * (-alpha * b^4 + (b^3) * beta + b * delta - epsilon - (b^2 * gamma))
    s2 = (1 / a) * (2 * alpha * b^3 - (3 * (b^2) * beta / 2) - (delta / 2) + b * gamma)

    if scale_coef
        phi_x = get_phi(coef, mesh_x)
        x_min, x_max = mesh_x.x_min, mesh_x.x_max
        phi_int = mesh_x.dx * vec(sum(phi_x, dims = 2)) ./ (x_max - x_min)
        phi_int = exp(a * phi_int)
        return (r1 * phi_int^2, r2 * phi_int, s1 / (phi_int^2), s2 / phi_int)
    else
        return r1, r2, s1, s2
    end

end


function compute_fe(coef, mesh_x, mesh_v, exp_a_phi_x, scale_coef = false)

    v = mesh_v.x
    a = coef.a

    exp_a_v2_div_2 = exp.(a .* (v .^ 2 ./ 2))'
    exp_a_v2 = exp.(a .* v .^ 2)'

    r1, r2, s1, s2 = get_ode_coef(coef, mesh_x, scale_coef)

    return sqrt.(-a / pi) .* (
        s1 .* (exp_a_phi_x .^ (-2)) .* exp_a_v2 .+
        (s2 ./ sqrt(2)) .* (exp_a_phi_x .^ (-1)) .* exp_a_v2_div_2
    )
end

function fe_init(coef, mesh_x, mesh_v)

    exp_a_phi_x = get_exp_a_phi(coef, mesh_x)
    compute_fe(coef, mesh_x, mesh_v, exp_a_phi_x)

end


function compute_fi(coef, mesh_x, mesh_v, exp_a_phi_x, scale_coef = false)

    a = coef.a
    v = mesh_v.x

    r1, r2, s1, s2 = get_ode_coef(coef, mesh_x, scale_coef)

    exp_a_v2_div_2 = exp.(a .* (v .^ 2 ./ 2))'
    exp_a_v2 = exp.(a .* v .^ 2)'

    return sqrt.(-a ./ pi) .* (
        r1 .* (exp_a_phi_x .^ 2) .* exp_a_v2 .+
        (r2 ./ sqrt(2)) .* exp_a_phi_x .* exp_a_v2_div_2
    )
end

function fi_init(coef, mesh_x, mesh_v)

    exp_a_phi_x = get_exp_a_phi(coef, mesh_x)
    compute_fi(coef, mesh_x, mesh_v, exp_a_phi_x)

end

function get_dx_exp_a_phi(coef, mesh_x)

    x = mesh_x.x
    lambda_coef = coef.lambda
    c = coef.c
    m = coef.m
    x0 = coef.x0
    sn, cn, dn = ellipj(lambda_coef .* x .+ x0, m)

    if coef.solution == :JacobiDN
        return -m .* c .* sn .* cn
    elseif coef.solution == :JacobiND
        return m .* c .* sn .* cn ./ (dn .^ 2)
    elseif coef.solution == :JacobiCN
        return -c .* sn .* dn
    else
        @error("The solution $(coef.solution) does not exist.")
    end

end

function get_dx_phi(coef, mesh_x)

    a = coef.a
    exp_a_phi = get_exp_a_phi(coef, mesh_x)
    dx_exp_a_phi = get_dx_exp_a_phi(coef, mesh_x)

    return (1 ./ a) .* dx_exp_a_phi ./ exp_a_phi

end

function compute_fe_prime(coef, mesh_x, mesh_v, exp_a_phi_x, dx_phi_x, scale_coef = false)

    v = mesh_v.x
    a = coef.a

    r1, r2, s1, s2 = get_ode_coef(coef, mesh_x, scale_coef)

    exp_a_v2_div_2 = exp.(a .* (v .^ 2 ./ 2))'
    exp_a_v2 = exp.(a .* v .^ 2)'

    f_eq_prime =
        sqrt.(-a / pi) .* (
            2 .* a .* s1 .* (exp_a_phi_x .^ (-2)) .* exp_a_v2 .+
            a .* (s2 ./ sqrt(2)) .* (exp_a_phi_x .^ (-1)) .* exp_a_v2_div_2
        )

    dv_f_eq = zero(f_eq_prime)
    dx_f_eq = zero(f_eq_prime)

    dv_f_eq = ein"ij,j->ij"(f_eq_prime, v)

    dx_f_eq = ein"ij,i->ij"(f_eq_prime, -dx_phi_x)

    return dx_f_eq, dv_f_eq

end

function fe_prime_init(coef, mesh_x, mesh_v)

    dx_phi = get_dx_phi(coef, mesh_x)
    exp_a_phi_x = get_exp_a_phi(coef, mesh_x)
    compute_fe_prime(coef, mesh_x, mesh_v, exp_a_phi_x, dx_phi)

end

function compute_fi_prime(coef, mesh_x, mesh_v, exp_a_phi_x, dx_phi_x, scale_coef = false)

    v = mesh_v.x

    a = coef.a
    r1, r2, s1, s2 = get_ode_coef(coef, mesh_x, scale_coef)

    exp_a_v2_div_2 = exp.(a .* (v .^ 2 ./ 2))'
    exp_a_v2 = exp.(a .* v .^ 2)'

    f_eq_prime =
        sqrt.(-a ./ pi) .* (
            2 .* a .* r1 .* (exp_a_phi_x .^ 2) .* exp_a_v2 .+
            a .* (r2 ./ sqrt(2)) .* exp_a_phi_x .* exp_a_v2_div_2
        )

    dv_f_eq = similar(f_eq_prime)
    dx_f_eq = similar(f_eq_prime)

    dv_f_eq = ein"ij,j->ij"(f_eq_prime, v)
    dx_f_eq = ein"ij,i->ij"(f_eq_prime, dx_phi_x)

    return dx_f_eq, dv_f_eq

end


function fi_prime_init(coef, mesh_x, mesh_v)

    dx_phi = get_dx_phi(coef, mesh_x)
    exp_a_phi_x = get_exp_a_phi(coef, mesh_x)
    compute_fi_prime(coef, mesh_x, mesh_v, exp_a_phi_x, dx_phi)

end
