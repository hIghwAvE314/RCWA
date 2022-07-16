from Params import *
from layers import *


""" Set up RCWA parameters """
params = RCWAParams()
params.dx, params.dy = 1e-3, 1e-3
params.Nmx, params.Nmy = 41, 41
params.acc=14
params.init()


""" Set up source parameters """
source = Source()
source.wl = 1.064
source.init(params)


""" Set up structure parameters """
class Geom(Structure):
    l1, gap, l2 = 0.40, 0.05, 0.27
    h = 0.46  # thickness of the main device
    w = 0.20  # width of the main device
    nl, nh = 1.5+0j, 3.45+0j  # refractive index of low RI and high RI medium
    Lx, Ly = 0.95, 0.60  # periodicity
    hsub = 0.4  # thickness of substrate
    hcap = 1 - h - hsub  # thickness of cap
    nrf = 1.33+0j
    ntm = 1.33+0j

    def init(self, params:RCWAParams):
        self.period = (self.Lx, self.Ly)
        self.Nx, self.Ny = int(1/params.dx+1), int(1/params.dy+1)
        self.x, self.y = np.mgrid[0:self.Lx:1j*self.Nx, 0:self.Ly:1j*self.Ny]
        mask = ((self.x>=0)*(self.x<=self.l1) + (self.x>=self.l1+self.gap)*(self.x<=self.l1+self.gap+self.l2)) * ((self.y>=self.w)*(self.y<=2*self.w))
        self.ur = [1.+0j, 1.+0j, 1.+0j]
        self.er = [self.nl**2]
        self.hs = [self.hsub]

        eps = np.where(mask, self.nh**2, self.nl**2)
        self.er.append(eps)
        self.hs.append(self.h)

        self.er.append(self.nl**2)
        self.hs.append(self.hcap)

        self.errf, self.urrf = self.nrf**2, 1.+0j
        self.ertm, self.urtm = self.ntm**2, 1.+0j

# class Geom(Structure):
#     nl, nh = 1.1+0j, 2.1+0j  # refractive index of low RI and high RI medium
#     Lx, Ly = 5, 5  # periodicity
#     h = 1.064/2
#     nrf = 1.1+0j
#     ntm = 1.1+0j

#     def init(self, params:RCWAParams):
#         self.period = (self.Lx, self.Ly)
#         self.Nx, self.Ny = int(self.Lx/params.dx+1), int(self.Ly/params.dy+1)
#         self.x, self.y = np.mgrid[0:self.Lx:1j*self.Nx, 0:self.Ly:1j*self.Ny]
#         mask = ((self.x>=self.Lx/4) * (self.x<=3*self.Lx/4))
#         self.ur = [1.+0j]
#         self.er = []
#         self.hs = []

#         eps = np.where(mask, self.nh**2, self.nl**2)
#         self.er.append(eps)
#         self.hs.append(self.h)

#         self.errf, self.urrf = self.nrf**2, 1.+0j
#         self.ertm, self.urtm = self.ntm**2, 1.+0j

geom = Geom()
geom.init(params)


"""Set up and run simulation"""
sim = Layers(params, source, geom)
sim.solve()
# R, T = sim.converge_test(81, step=4, comp='xy')
# plt.plot(R)
# plt.plot(T)
# plt.show()

plt.imshow(np.real(sim.Ref))
plt.show()
plt.imshow(np.real(sim.Trm))
plt.show()
plt.scatter(sim.params.mx, np.real(sim.Ref[:, sim.params.My]))
plt.scatter(sim.params.mx, np.real(sim.Trm[:, sim.params.My]))
plt.show()

for n, m in enumerate(sim.params.mx):
    r = np.real(sim.Ref[n, sim.params.My])
    t = np.real(sim.Trm[n, sim.params.My])
    if not np.isclose(t, 0): print(f"Transmittance: {m}, {t}") 
    if not np.isclose(r, 0): print(f"Reflectance: {m}, {r}")

print(sim.Rtot + sim.Ttot)
