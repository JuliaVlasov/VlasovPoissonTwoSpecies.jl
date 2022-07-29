import numpy as np 

class Mesh:
	def __init__(self, x_min, x_max, nx):
		self.x     = np.linspace(x_min, x_max, nx, endpoint=False)
		self.x_min = x_min
		self.x_max = x_max
		self.nx    = nx
		self.dx    = self.x[1]-self.x[0]