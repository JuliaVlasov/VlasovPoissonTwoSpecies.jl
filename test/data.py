import numpy as np

class Data:
	def __init__(self):
		self.T_final           = 1
		self.nb_time_steps     = 10
		self.nx                = 64
		self.nv                = 64
		self.x_min             = 0
		self.x_max             = 1
		self.v_min             = -10
		self.v_max             = 10
		self.output            = True
		self.vtk               = True
		self.freq_output       = 1
		self.freq_save         = 1
		self.wb_scheme         = False
		self.coef              = {"solution type": "JacobiDN", "lambda": 1., "a": -1.,
								  "b": 1 + np.sqrt(2) + 0.12, "c": 1.,"m": 0.97,"x0": 0.}
		self.perturbation_init = lambda x, v: 0.
		self.projection        = False
		self.projection_type   = "coefficients"
		self.freq_projection   = 1
