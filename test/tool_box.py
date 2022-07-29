import numpy as np


def integrate(*args):
    if len(args) == 3:
        return integrate_2D(args[0], args[1], args[2])
    elif len(args) == 2:
        return integrate_1D(args[0], args[1])
    else:
        raise ValueError(f"Wrong number of argument in function integrate (should be 2 or 3) but got {len(args)} arguments.")


def integrate_2D(mesh_x, mesh_v, f):
    assert((f.shape[-2], f.shape[-1]) == (mesh_x.nx, mesh_v.nx)
           and len(f.shape) in (2,3)), \
          (f"Can't integrate function f"
           f"with shape {f.shape}")

    dx = mesh_x.dx
    dv = mesh_v.dx
    return dv * dx * f.sum((-2, -1))

def integrate_1D(mesh_x, f):
    assert((f.shape[-1]) == (mesh_x.nx)
           and len(f.shape) in (1,2)), \
          (f"Can't integrate function f"
           f"with shape {f.shape}")

    dx = mesh_x.dx
    return dx * f.sum(-1)

#def integrate(mesh_x, mesh_v, f):
#    assert(f.shape == (mesh_x.nx, mesh_v.nx)), \
#          (f"Can't integrate numpy function f"
#           f"with shape {f.shape}")
#
#    dx = mesh_x.dx
#    dv = mesh_v.dx
#    return dv * dx * np.sum(f)


def compute_rho(mesh_v, f):
    if len(f.shape) == 4:
        assert f.shape[3] == 1, f"Can't compute rho for function with shape {f.shape}"
        f = f.reshape(f.shape[0:-1])
    else:
        assert len(f.shape) in (2, 3), f"Can't compute rho for function with shape {f.shape}"

    assert(f.shape[-1] == mesh_v.nx), \
          (f"Can't compute_rho for function with shape {f.shape}")

    dv = mesh_v.dx
    return dv * f.sum(-1)


" Compute dx Ex = rho using that ik*Ex = rho "
def compute_e(mesh_x, rho):
    assert(rho.shape[-1] == mesh_x.nx), \
          (f"Can't compute_e with rho with"
           f"with shape {rho.shape}")

    nx = mesh_x.nx
    k = 2 * np.pi / (mesh_x.x_max - mesh_x.x_min)
    modes = k * np.fft.fftfreq(nx, 1 / nx)
    modes[0] = 1.0

    fft_rho = np.fft.fft(rho)
    fft_rho[0] = 0.
    rhok = fft_rho / modes
    rhok *= -1j

    return (np.fft.ifft(rhok)).real


def compute_phi(mesh_x, rho):
    nx = mesh_x.nx
    k = 2 * np.pi / (mesh_x.x_max - mesh_x.x_min)
    modes = k * np.fft.fftfreq(nx, 1 / nx)
    modes[0] = 1.0

    fft_rho = np.fft.fft(rho)
    fft_rho[0] = 0.
    rhok = fft_rho / (- modes**2)

    return (np.fft.ifft(rhok)).real


def get_df_FD_matrix(size_f, order=8):
    A = np.zeros([size_f, size_f])

    if order == 2:
        coef_right = [0.5]
    elif order == 4:
        coef_right = [2 / 3, -1 / 12]
    elif order == 6:
        coef_right = [3 / 4, -3 / 20, 1 / 60]
    elif order == 8:
        coef_right = [4 / 5, -1 / 5, 4 / 105, -1 / 280]
    else:
        raise ValueError(f"The order {order} is not implemented for centered finite difference.")

    right_pos = [i + 1 for i in range(0, len(coef_right))]

    for i in range(0, size_f):
        # current_right_pos = ((right_pos .+ (i-1)).%(length(f))).+1a
        current_right_pos = [i + pos for pos in right_pos]
        current_right_pos = [p if p < size_f else p -
                             size_f for p in current_right_pos]

        current_left_pos = [i - pos for pos in right_pos]
        current_left_pos = [p if p >= 0 else size_f +
                            p for p in current_left_pos]

        for j in range(0, len(coef_right)):
            A[i, current_right_pos[j]] += coef_right[j]
            A[i, current_left_pos[j]] += -coef_right[j]

    return A


def df_centered(f, dx, A=None, order=8):
    """
    Calculate the centered first derivative of f at a given order.
    """

    if A is None:
        A = get_df_FD_matrix(int(f.shape[0]), order)

    shape_1D = False
    if len(f.shape) == 1:
        shape_1D = True
        f = f.reshape([f.shape[0], 1])

    df = np.matmul(A, f)

    df = df / (dx)
    if shape_1D:
        return df.reshape(-1)
    else:
        return df


def LP_norm(*args):
    """
    General function to calulate LP norm.
    For the 1D case the arguments should be (mesh_x, f, p).
    For the 2D case the arguments should be (mesh_x, mesh_v, f, p).
    """
    # 2D case
    if len(args) == 4:
        return LP_norm_2D(args[0], args[1], args[2], args[3])
    # 1D case
    elif len(args) == 3:
        return LP_norm_1D(args[0], args[1], args[2])
    else:
        raise ValueError(f"Wrong number of argument in the function LP_norm (should be 3 or 4) but got {len(args)} arguments.")


def LP_norm_2D(mesh_x, mesh_v, f, p):
    assert (f.shape == (mesh_x.nx, mesh_v.nx)), \
           (f"Can't evaluate the 2D LP norm of a numpy function "
            f"with shape {f.shape}")

    dv = mesh_v.dx
    dx = mesh_x.dx

    return np.sum(dx * dv * np.abs(f)**p)**(1 / p)


def LP_norm_1D(mesh_x, f, p):
    assert(f.shape == (mesh_x.nx,)), \
          (f"Can't evaluate the 1D LP norm of a numpy function "
           f"with shape {f.shape}")

    dx = mesh_x.dx
    return np.sum(dx * np.abs(f)**p)**(1 / p)


def L2_norm(*args):
    return LP_norm(*args, 2)

def compute_dx_f(mesh, f, order=8):
    A_dx = get_df_FD_matrix(mesh.nx, order)
    if len(f.shape) == 2:
        dx_f = np.einsum("ij,jl->il", A_dx, f) / mesh.dx
    elif len(f.shape) == 1:
        dx_f = np.einsum("ij,j->i", A_dx, f) / mesh.dx
    else:
        raise ValueError(f"Can't evaluate the derivative of function with shape {f.shape}.")

    return dx_f

def T_f(mesh_x, mesh_v, f, e, dx_f=None, dv_f=None,
        A_dx=None, A_dv=None, order=8):
    """
    Return (v*dx+e*dv) f
    """
    assert len(f.shape) == 2, \
        f"Can't evaluate T_f of a tensor function with shape {f.shape}"
    assert f.shape == (mesh_x.nx, mesh_v.nx), \
        f"Can't evaluate T_f of a tensor function with shape {f.shape}"

    dx = mesh_x.dx
    dv = mesh_v.dx
    x = mesh_x.x
    v = mesh_v.x

    if A_dx is None:
        A_dx = get_df_FD_matrix(mesh_x.nx, order)
    if A_dv is None:
        A_dv = get_df_FD_matrix(mesh_v.nx, order)

    if dx_f is None:
        v_dx_f = np.einsum("ij,jl->il", A_dx, f) / dx
        v_dx_f = np.einsum("jk,k->jk", v_dx_f, v)
    else:
        v_dx_f = np.einsum("jk,k->jk", dx_f, v)

    if dv_f is None:
        e_dv_f = np.einsum("jk,lk->jl", f, A_dv) / dv
        e_dv_f = np.einsum("jk,j->jk", e_dv_f, e)
    else:
        e_dv_f = np.einsum("jk,j->jk", dv_f, e)


    return v_dx_f + e_dv_f
