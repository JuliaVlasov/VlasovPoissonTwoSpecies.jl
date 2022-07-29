import numpy as np

def compute_energy(mesh_x, mesh_v, f):
    """
    Return the energy e_f = int v^2 f dv
    """
    v  = mesh_v.x
    dx = mesh_x.dx
    dv = mesh_v.dx
    energy = v**2 * f

    return dv * dx * np.sum(energy)


def get_normalized_energy(mesh_x, mesh_v, f, energy_eq):
    """
    Return the normalized energy |e_f - e_eq| / e_eq
    where e_f is the energy of f.
    """
    return np.abs(compute_energy(mesh_x, mesh_v, f) - energy_eq) / energy_eq


class OutputManager:
    def __init__(self, data, mesh_x, mesh_v, fe_eq_init, fi_eq_init):
        self.data              = data
        self.mesh_x            = mesh_x
        self.mesh_v            = mesh_v
        self.nb_outputs        = 0
        self.t                 = []
        self.energy_fe_eq_init = compute_energy(mesh_x, mesh_v, fe_eq_init)
        self.energy_fi_eq_init = compute_energy(mesh_x, mesh_v, fi_eq_init)
        self.compute_energy_eq = False
        self.energy_fe         = []
        self.energy_fi         = []

    def save_output(self, fe, fi, fe_eq, fi_eq, t):
        data   = self.data
        mesh_x = self.mesh_x
        mesh_v = self.mesh_v

        self.t.append(t)
        e_fe, e_fi = self.compute_normalized_energy(fe, fi)
        self.energy_fe.append(e_fe)
        self.energy_fi.append(e_fi)

    def compute_normalized_energy(self, fe, fi):
        mesh_x = self.mesh_x
        mesh_v = self.mesh_v

        return (get_normalized_energy(mesh_x, mesh_v, fe, self.energy_fe_eq_init),
                get_normalized_energy(mesh_x, mesh_v, fi, self.energy_fi_eq_init))

    def end_of_simulation(self):
        energy_fe = np.array([self.t, self.energy_fe])
        energy_fi = np.array([self.t, self.energy_fi])

        np.save("output/energy_fe.npy", energy_fe)
        np.save("output/energy_fi.npy", energy_fi)

