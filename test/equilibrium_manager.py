import numpy as np
from tool_box import integrate
from scipy.special import ellipj


class EquilibriumManager:
    def __init__(self, mesh_x, mesh_v, coef, init_eq=True):
        self.mesh_x = mesh_x
        self.mesh_v = mesh_v
        self.coef   = coef
        self.solution_type = coef.get("solution type", None)
        if self.solution_type is None:
            raise Exception("Error the solution type must be specified.")

        self.coef_init()
        if init_eq:
            self.Fe_init()
            self.Fi_init()
            self.Fe_prime_init()
            self.Fi_prime_init()
        print(self.fe_eq.shape)

    def get_phi(self):
        a = self.coef.get("a", -1.)
        return np.log(self.get_exp_a_phi()**(1 / a))

    def get_dx_phi(self):
        coef = self.coef
        a = coef.get("a", -1.)

        exp_a_phi = self.get_exp_a_phi()
        dx_exp_a_phi = self.get_dx_exp_a_phi()

        return (1. / a) * dx_exp_a_phi / exp_a_phi

    def get_exp_a_phi(self):
        coef = self.coef
        x    = self.mesh_x.x
        lambda_coef = coef.get("lambda", 1.)
        b = coef.get("b", 2.)
        c = coef.get("c", 1.)
        m = coef.get("m", 0.5)
        x0 = coef.get("x0", 0.)
        elliptic_f = ellipj(lambda_coef * x + x0, m)
        cn = elliptic_f[1]
        dn = elliptic_f[2]

        if self.solution_type == "JacobiDN":
            return (c * dn + b)
        elif self.solution_type == "JacobiND":
            return (c / dn) + b
        elif self.solution_type == "JacobiCN":
            return (c * cn + b)
        else:
            raise Exception(f"The solution {self.solution_type} does not exist.")

    def get_dx_exp_a_phi(self):
        coef = self.coef
        x    = self.mesh_x.x
        lambda_coef = coef.get("lambda", 1.)
        c = coef.get("c", 1.)
        m = coef.get("m", 0.5)
        x0 = coef.get("x0", 0.)
        elliptic_f = ellipj(lambda_coef * x + x0, m)
        sn = elliptic_f[0]
        cn = elliptic_f[1]
        dn = elliptic_f[2]

        if self.solution_type == "JacobiDN":
            return - m * c * sn * cn
        elif self.solution_type == "JacobiND":
            return m * c * sn * cn / (dn**2)
        elif self.solution_type == "JacobiCN":
            return - c * sn * dn
        else:
            raise Exception(f"The solution {self.solution_type} does not exist.")

    def coef_init(self):
        coef          = self.coef
        solution_type = self.solution_type
        lambda_coef   = coef.get("lambda", 1.)
        c = coef.get("c", 1.)
        m = coef.get("m", 0.5)

        if solution_type == "JacobiDN":
            alpha  = -lambda_coef**2 / (c**2)
            beta   = 0.
            gamma  = (lambda_coef**2) * (2 - m)
            delta  = 0.
            epsilon = -(lambda_coef**2) * (1 - m) * c**2
        elif solution_type == "JacobiND":
            alpha  = (lambda_coef**2) * (m - 1) / (c**2)
            beta   = 0.
            gamma  = (lambda_coef**2) * (2 - m)
            delta  = 0.
            epsilon = -(lambda_coef**2) * c**2
        elif solution_type == "JacobiCN":
            alpha  = - m * (lambda_coef**2) / (c**2)
            beta   = 0.
            gamma  = (lambda_coef**2) * (2 * m - 1)
            delta  = 0.
            epsilon = (lambda_coef**2) * (1 - m) * c**2
        else:
            raise Exception('Solution type {} not defined'.format(solution_type))

        coef["alpha"] = alpha
        coef["beta"] = beta
        coef["gamma"] = gamma
        coef["delta"] = delta
        coef["epsilon"] = epsilon

    def check_coef(self):
        coef = self.coef
        return all(k in coef for k in ("alpha", "beta", "gamma", "delta", "epsilon"))

    def get_ode_coef(self, scale_coef=False):
        if not self.check_coef():
            raise Exception("Coefficients are not well defined.")

        coef = self.coef
        a    = coef.get("a", -1.)
        b = coef.get("b", 2.)

        alpha = coef["alpha"]
        beta  = coef["beta"]
        gamma = coef["gamma"]
        delta = coef["delta"]
        epsilon = coef["epsilon"]
        r1 = - alpha / a
        r2 = - (-2 * alpha * b + (beta / 2)) / a
        s1 = (1 / a) * (-alpha * b**4 + (b**3) * beta + b * delta - epsilon - (b**2 * gamma))
        s2 = (1 / a) * (2 * alpha * b**3 - (3 * (b**2) * beta / 2) - (delta / 2) + b * gamma)

        if scale_coef:
            phi_x = self.get_phi()
            x_min, x_max = self.mesh_x.x_min, self.mesh_x.x_max
            phi_int = integrate(self.mesh_x, phi_x) / (x_max - x_min)
            phi_int = np.exp(a * phi_int)
            return (r1 * phi_int**2, r2 * phi_int,
                s1 / (phi_int**2), s2 / phi_int)
        else:
            return r1, r2, s1, s2

    def compute_Fe(self, exp_a_phi_x, scale_coef=False):
        if not self.check_coef():
            raise Exception("Coefficients are not well defined.")

        coef = self.coef
        v    = self.mesh_v.x

        a = coef.get("a", -1.)

        exp_a_v2_div_2 = np.exp(a * (v**2 / 2))
        exp_a_v2 = np.exp(a * v**2)

        *_, s1, s2 = self.get_ode_coef(scale_coef)

        print(exp_a_phi_x.shape)
        print(exp_a_phi_x[0])
        print(s1, s2)
        print(exp_a_v2_div_2.shape)
        print(exp_a_v2.shape)
        print(a, s1, s2)

        fe = np.array([np.sqrt(-a / np.pi) * (s1 * (exp_a_phi_xi**(-2))
            * exp_a_v2 + (s2 / np.sqrt(2)) * (exp_a_phi_xi**(-1))
            * exp_a_v2_div_2) for exp_a_phi_xi in exp_a_phi_x])
        print(fe.shape)
        return fe


    def compute_Fi(self, exp_a_phi_x, scale_coef=False):
        if not self.check_coef():
            raise Exception("Coefficients are not well defined.")

        coef = self.coef
        a    = coef.get("a", -1.)
        v    = self.mesh_v.x

        r1, r2, *_ = self.get_ode_coef(scale_coef)

        exp_a_v2_div_2 = np.exp(a * (v**2 / 2))
        exp_a_v2 = np.exp(a * v**2)

        return np.array([np.sqrt(-a / np.pi) * (r1 * (exp_a_phi_xi**2)
            * exp_a_v2 + (r2 / np.sqrt(2)) * exp_a_phi_xi
            * exp_a_v2_div_2) for exp_a_phi_xi in exp_a_phi_x])


    def compute_Fe_prime(self, exp_a_phi_x, dx_phi_x, scale_coef=False):
        if not self.check_coef():
            raise Exception("Coefficients are not well defined.")

        coef = self.coef
        v    = self.mesh_v.x

        a = coef.get("a", -1.)
        *_, s1, s2 = self.get_ode_coef(scale_coef)

        # exp_a_phi_x = np.exp(a * phi_x)
        exp_a_v2_div_2 = np.exp(a * (v**2 / 2))
        exp_a_v2 = np.exp(a * v**2)

        fi_eq_prime = np.array([np.sqrt(-a / np.pi) * (2 * a * s1 * (
            exp_a_phi_xi**(-2)) * exp_a_v2 + a * (
            s2 / np.sqrt(2)) * (exp_a_phi_xi**(-1)) * exp_a_v2_div_2)
            for exp_a_phi_xi in exp_a_phi_x])
        dv_fi_eq = np.einsum("ij,j->ij", fi_eq_prime, v)

        dx_fi_eq = np.einsum("ij,i->ij", fi_eq_prime, -dx_phi_x)

        return dx_fi_eq, dv_fi_eq


    def compute_Fi_prime(self, exp_a_phi_x, dx_phi_x, scale_coef=False):
        if not self.check_coef():
            raise Exception("Coefficients are not well defined.")

        coef = self.coef
        v    = self.mesh_v.x

        a = coef.get("a", -1.)
        r1, r2, *_ = self.get_ode_coef(scale_coef)

        exp_a_v2_div_2 = np.exp(a * (v**2 / 2))
        exp_a_v2 = np.exp(a * v**2)

        fe_eq_prime = np.array([np.sqrt(-a / np.pi) * (2 * a * r1 * (
            exp_a_phi_xi**2) * exp_a_v2 + a * (
            r2 / np.sqrt(2)) * exp_a_phi_xi * exp_a_v2_div_2)
            for exp_a_phi_xi in exp_a_phi_x])
        dv_fe_eq = np.einsum("ij,j->ij", fe_eq_prime, v)

        dx_fe_eq = np.einsum("ij,i->ij", fe_eq_prime, dx_phi_x)

        return dx_fe_eq, dv_fe_eq

    def Fe_init(self):
        exp_a_phi_x = self.get_exp_a_phi()
        self.fe_eq = self.compute_Fe(exp_a_phi_x)

    def Fi_init(self):
        exp_a_phi_x = self.get_exp_a_phi()
        self.fi_eq = self.compute_Fi(exp_a_phi_x)

    def Fe_prime_init(self):
        dx_phi = self.get_dx_phi()
        exp_a_phi_x  = self.get_exp_a_phi()
        self.dx_fe_eq, self.dv_fe_eq = self.compute_Fe_prime(exp_a_phi_x, dx_phi)

    def Fi_prime_init(self):
        # phi_x  = self.get_phi()
        dx_phi = self.get_dx_phi()
        exp_a_phi_x  = self.get_exp_a_phi()
        self.dx_fi_eq, self.dv_fi_eq = self.compute_Fi_prime(exp_a_phi_x, dx_phi)
