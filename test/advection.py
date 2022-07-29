import numpy as np
import multiprocessing as mp
import numpy as np

def bspline(p, j, x):
	"""
	Return the value at x in [0,1[ of the B-spline with
	integer nodes of degree p with support starting at j.
	Implemented recursively using the de Boor's recursion formula
	"""
	if p == 0:
		if j == 0:
			return 1.0
		else:
			return 0.0
	else:
		w = (x - j) / p
		w1 = (x - j - 1) / p
	return (w * bspline(p - 1, j, x) + (1 - w1) * bspline(p - 1, j + 1, x))


class Advection:
	""" Advection to be computed on each row """
	def __init__(self, mesh, p=3):
		self.mesh = mesh
		self.p     = p
		nx        = mesh.nx
		#modes    = np.zeros(nx, np.float64)
		self.modes    = np.array([2*np.pi * i / nx for i in range(0,nx)])
		self.eig_bspl = np.zeros(nx, np.float64)
		self.eig_bspl.fill(bspline(p, -np.floor((p+1)/2), 0.0))
		for i in range(1,int(np.floor(p+1/2))):
			self.eig_bspl += bspline(p, i-np.floor((p+1)/2), 0.0) * 2 * np.cos(i * self.modes)

		self.eigalpha  = np.zeros(nx, np.complex128)

	def advect_iteration(self, j):
		f_fft = self.f_fft
		v = self.v
		dt = self.dt
		nx = self.mesh.nx
		dx = self.mesh.dx
		alpha = dt * v[j] / dx
		# compute eigenvalues of cubic splines evaluated at displaced points
		ishift = np.floor(-alpha)
		beta   = -ishift - alpha
		eigalpha  = np.zeros(nx, np.complex128)

		for i in range(-int(np.floor((self.p - 1) / 2)), int(np.floor((self.p + 1) / 2) + 1)):
				eigalpha += (bspline(self.p, i - np.floor((self.p + 1) / 2), beta)
				* np.exp((ishift + i) * 1j * self.modes))

		f_fft[j, :] *= self.eigalpha / self.eig_bspl
		return (j, np.real(np.fft.ifft(f_fft[j, :] * eigalpha / self.eig_bspl)))

	def advect(self, f, v, dt):
		nx = self.mesh.nx
		nv = len(v)
		dx = self.mesh.dx

		f_fft = np.fft.fft(f, axis=1)

		#Parallel(n_jobs=1)(delayed(self.advect_iteration)(f_fft, v, j, dt) for j in range(0,nv))

		# with mp.Pool(processes=2) as pool:
		# 	self.f_fft = f_fft
		# 	self.v = v
		# 	self.dt = dt
		# 	result = pool.map(self.advect_iteration, range(nv))
		# for row in result:
		# 	i = row[0]
		# 	f[i, :] = row[1]

		for j in range(0, nv):
			alpha = dt * v[j] / dx
			# compute eigenvalues of cubic splines evaluated at displaced points
			ishift = np.floor(-alpha)
			beta   = -ishift - alpha
			self.eigalpha.fill(0.0j)
			for i in range(-int(np.floor((self.p - 1) / 2)), int(np.floor((self.p + 1) / 2) + 1)):
				self.eigalpha += (bspline(self.p, i - np.floor((self.p + 1) / 2), beta)
					* np.exp((ishift + i) * 1j * self.modes))
			f_fft[j, :] *= self.eigalpha / self.eig_bspl

		f[:, :] = np.real(np.fft.ifft(f_fft, axis=1))


