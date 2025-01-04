from matplotlib.path import Path
import numpy as np

from rcwa.rcwa import Structure, RCWAParams


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

class RoundedTriangle(Structure):
    Lx, Ly = 0.95, 0.6  # Periodicity
    wx, wy = 0.8, 0.5  # triangle size in y and x direction
    nl, nh = 1.45-0j, 3.45-0j  # refractive index of low RI and high RI medium
    h = 0.46  # thickness of the main device
    hsub = 0.4  # thickness of substrate
    htot = 1
    nrf = 1.33-0j
    ntm = 1.33-0j
    radius = 0.01  # radius of rounded corner

    def _get_corner(self):
        lx = -self.wx / 2
        rx = -lx  # x of right conners
        y = self.wy / 2  # abs(y) of the top and bottom corners, the other one y=0
        return lx, rx, y
    
    def _get_circle_center(self):
        area = self.wy*self.wx
        side_a = self.wy # side length type a
        side_b = np.sqrt((self.wy/2)**2 + self.wx**2)  # side length type b
        bc_a = self.radius * side_a / area  # barycentric coordinate type a
        bc_b = self.radius * side_b / area  # barycentric coordinate type b
        assert 2*bc_b < 1, "radius too large"
        assert bc_a*bc_b < 1, "radius too large"
#         v1x = bc_b * (-self.wx/2) + bc_b * (-self.wx/2) + (1-2*bc_b) * (self.wx/2)
        v1x = (1 - 4*bc_b)*self.wx/2
        v1y = 0
#         v2x = bc_a * (self.wx/2) + bc_b*(-self.wx/2) + (1-bc_a-bc_b)*(-self.wx/2)
        v2x = (2*bc_a - 1)*self.wx/2
#         v2y = bc_a * (0) + bc_b*(-self.wy/2) + (1-bc_a-bc_b)*(self.wy/2)
        v2y = (1 - bc_a - 2*bc_b)*self.wy/2
        v3x = v2x
        v3y = -v2y
        return v1x, v1y, v2x, v2y, v3x, v3y
    
    def get_mask(self, X, Y):
        lx, rx, y = self._get_corner()
        x1, y1 = rx, 0
        x2, y2 = lx, y
        x3, y3 = lx, -y
        
        points = np.transpose([X.ravel(), Y.ravel()])
        tri = Path([[x1, y1], [x2, y2], [x3, y3]])
        tcond = tri.contains_points(points).reshape(X.shape) 
        
        v1x, v1y, v2x, v2y, v3x, v3y = self._get_circle_center()
        vcond1 = (np.sqrt((X-x1)**2 + (Y-y1)**2)<=np.sqrt((v1x-x1)**2+(v1y-y1)**2)) * (np.sqrt((X-v1x)**2 + (Y-v1y)**2)>=self.radius)
        vcond2 = (np.sqrt((X-x2)**2 + (Y-y2)**2)<=np.sqrt((v2x-x2)**2+(v2y-y2)**2)) * (np.sqrt((X-v2x)**2 + (Y-v2y)**2)>=self.radius)
        vcond3 = (np.sqrt((X-x3)**2 + (Y-y3)**2)<=np.sqrt((v3x-x3)**2+(v3y-y3)**2)) * (np.sqrt((X-v3x)**2 + (Y-v3y)**2)>=self.radius)
        
#         return tcond (vcond1 + vcond2 + vcond3)
        return 1*tcond - (vcond1 + vcond2 + vcond3)*tcond

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