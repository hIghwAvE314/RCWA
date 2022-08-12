from Params import *
from layers import *
import tracemalloc


tracemalloc.start()

""" Set up RCWA parameters """
params = RCWAParams()
params.dx, params.dy = 1e-3, 1e-3
params.Nmx, params.Nmy = 41, 41
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
cpu, cpu_peak = tracemalloc.get_traced_memory()
print(f"cpu mem usage of simulation initialization: {cpu/1024**2}MB with peak {cpu_peak/1024**2}MB")
sim.solve()
# R, T = sim.converge_test(161, step=8, comp='x', acc=1e-6)
# plt.plot(R)
# plt.plot(T)
# plt.show()

# kx = sim.K.Kx.diagonal().reshape(sim.params.Nmx, sim.params.Nmy)
# ky = sim.K.Ky.diagonal().reshape(sim.params.Nmx, sim.params.Nmy)
# real_mode_mask = np.isclose(np.imag(kx),0) + np.isclose(np.imag(ky),0)
# plt.imshow(np.real(sim.Ref[real_mode_mask]))
# plt.show()
# plt.imshow(np.real(sim.Trm[real_mode_mask]))
# plt.show()

# plt.stem(sim.params.mx, np.real(sim.Ref[:, sim.params.My]), markerfmt='x', label='ref')
# plt.stem(sim.params.mx, np.real(sim.Trm[:, sim.params.My]), markerfmt='x', label='trm')
# plt.legend()
# plt.show()
F = sim.get_force('mode_data')

max_mem = torch.cuda.max_memory_allocated()
total_mem = torch.cuda.get_device_properties('cuda').total_memory
cpu, cpu_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nMaximum cuda memory usage: {max_mem/1024**2}MB ({max_mem/total_mem * 100 :.4f}%)")
print(f"cpu mem usage of total simulation: {cpu/1024**2}MB with peak {cpu_peak/1024**2}MB")

print("\nReflectance:")
for n, m in enumerate(sim.params.mx):
    r = np.real(sim.Ref[n, sim.params.My])
    if not np.isclose(r, 0): print(f"mode {m}: {r}")
print("\nTransmittance:")
for n, m in enumerate(sim.params.mx):
    t = np.real(sim.Trm[n, sim.params.My])
    if not np.isclose(t, 0): print(f"Transmittance: {m}, {t}") 

print("\nTotal Reflectance: ",sim.Rtot)
print("Total Transmittance: ",sim.Ttot)
print(f"Extinction: {1 - (sim.Rtot + sim.Ttot) :.4f}")

print(f"\nForce coefficient: {F}")
