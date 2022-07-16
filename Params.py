import numpy as np
from scipy import linalg as LA


class RCWAParams:
    dx, dy = 1e-3, 1e-3  # resolution in real space (um)
    Nmx, Nmy = 7, 7  # number of harmonics in each direction
    acc = 16

    is_init = False

    def init(self):
        self.Nmodes = self.Nmx * self.Nmy  # total number of hormonics
        self.Mx, self.My = self.Nmx//2, self.Nmy//2  # index of each mode
        self.mx, self.my = np.mgrid[-self.Mx:self.Mx+1], np.mgrid[-self.My:self.My+1]
        self.modes = np.meshgrid(self.mx, self.my, indexing='ij')
        self.modex = self.modes[0].reshape(self.Nmodes)
        self.modey = self.modes[1].reshape(self.Nmodes)

        self.is_init = True


class Source:
    wl = 1.  # wavelength in vacuum (um)
    theta = 0.  # azimutal angle of incident light, 0 for normal incident (degrees)
    phi = 0.  # polar angle of incident light, 0 for normal incident or positive x component (degrees)
    pol = np.array([1, 0, 0])  # the polarisation vector of E-field (normalised to 1)

    is_init = False

    def init(self, params:RCWAParams):
        """ninc: refractive index of incident side material"""
        self.k0 = 2*np.pi / self.wl
        th = np.deg2rad(self.theta)
        ph = np.deg2rad(self.phi)
        self.inc = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
        delta = np.where(np.isclose(params.modex, 0) * np.isclose(params.modey, 0), 1., 0.)
        self.pol = self.pol / LA.norm(self.pol)
        self.e_src = np.concatenate([self.pol[0]*delta, self.pol[1]*delta])

        self.is_init = True


class Structure:
    """
    This is the superclass for all geometry class, it should includes properties bellow after initialisation:
    errf, urrf, ertm, urtm : reflection (incident) side and transmission side permittivity and permeability
    er, ur, hs : list of permittivity, permeability and thickness for each layers in order
    Nx, Ny, x, y : number of points in real space and coordinates for each point if the layer is not homogeneous
    period : a tuple reperesents periodicity of the structure (Lx, Ly)
    """
    pass