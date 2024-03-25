from matplotlib.path import Path
import numpy as np

from RCWAv2 import *


class Triangle(Structure):
    Lx, Ly = 0.95, 0.6  # Periodicity
    wx, wy = 0.8, 0.5  # triangle size in y and x direction
    nl, nh = 1.45-0j, 3.45-0j  # refractive index of low RI and high RI medium
    h = 0.46  # thickness of the main device
    hsub = 0.4  # thickness of substrate
    htot = 1
    nrf = 1.33-0j
    ntm = 1.33-0j

    def _get_corner(self):
        lx = -self.wx / 2
        rx = -lx  # x of right conners
        y = self.wy / 2  # abs(y) of the top and bottom corners, the other one y=0
        return lx, rx, y

    def get_mask(self, X, Y):
        lx, rx, y = self._get_corner()
        x1, y1 = lx, y
        x2, y2 = lx, -y
        x3, y3 = rx, 0
        points = np.transpose([X.ravel(), Y.ravel()])
        tri = Path([[x1, y1], [x2, y2], [x3, y3]])
        tcond = tri.contains_points(points).reshape(X.shape) 
        return tcond

    def init(self, params:RCWAParams):
        self.hcap = self.htot - self.h - self.hsub  # thickness of cap
        self.period = (self.Lx, self.Ly)
        self.Nx, self.Ny = int(1/params.dx+1), int(1/params.dy+1)
        self.x, self.y = np.mgrid[-self.Lx/2:self.Lx/2:1j*self.Nx, -self.Ly/2:self.Ly/2:1j*self.Ny]
        self.ur = [1.+0j, 1.+0j, 1.+0j]
        self.er = [self.nl**2]
        self.hs = [self.hsub]

        mask = self.get_mask(self.x, self.y)
        eps = np.where(mask, self.nh**2, self.nl**2)
        self.er.append(eps)
        self.hs.append(self.h)

        self.er.append(self.nl**2)
        self.hs.append(self.hcap)

        self.errf, self.urrf = self.nrf**2, 1.+0j
        self.ertm, self.urtm = self.ntm**2, 1.+0j


class TriangleGap(Triangle):
    cx, cy = 0., 0.
    gap = .05
    def get_mask(self, X, Y):
        tcond = super().get_mask(X, Y) 
        gcond = ((X-self.cx)**2) > (self.gap/2)**2
        return tcond * gcond


class TriangleHole(Triangle):
    cx, cy = 0., 0.
    r = .05
    def get_mask(self, X, Y):
        tcond = super().get_mask(X, Y) 
        rcond = ((X-self.cx)**2 + (Y-self.cy)**2) > self.r**2
        return tcond * rcond


class MetaVehicle(Structure):
    l1, gap, l2 = 0.40, 0.05, 0.27
    h = 0.46  # thickness of the main device
    w = 0.20  # width of the main device
    nl, nh = 1.45-0j, 3.45-0j  # refractive index of low RI and high RI medium
    Lx, Ly = 0.95, 0.60  # periodicity
    hsub = 0.4  # thickness of substrate
    hcap = 1 - h - hsub  # thickness of cap
    nrf = 1.33-0j
    ntm = 1.33-0j

    def get_mask(self, X, Y):
        mask = ((X>=0)*(X<=self.l1) + (X>=self.l1+self.gap)*(X<=self.l1+self.gap+self.l2)) * ((Y>=self.w)*(Y<=2*self.w))
        return mask

    def init(self, params:RCWAParams):
        self.period = (self.Lx, self.Ly)
        self.Nx, self.Ny = int(1/params.dx+1), int(1/params.dy+1)
        self.x, self.y = np.mgrid[0:self.Lx:1j*self.Nx, 0:self.Ly:1j*self.Ny]
        mask = ((self.x>=0)*(self.x<=self.l1) + (self.x>=self.l1+self.gap)*(self.x<=self.l1+self.gap+self.l2)) * ((self.y>=self.w)*(self.y<=2*self.w))
        self.ur = [1.-0j, 1.-0j, 1.-0j]
        self.er = [self.nl**2]
        self.hs = [self.hsub]

        eps = np.where(mask, self.nh**2, self.nl**2)
        self.er.append(eps)
        self.hs.append(self.h)

        self.er.append(self.nl**2)
        self.hs.append(self.hcap)

        self.errf, self.urrf = self.nrf**2, 1.+0j
        self.ertm, self.urtm = self.ntm**2, 1.+0j