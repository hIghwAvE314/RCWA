from Params import *
from rcwa import *


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



def converge(nmode):
    params = RCWAParams()
    params.dx, params.dy = 1e-3, 1e-3
    params.Nmx, params.Nmy = 41, nmode
    params.init()

    source = Source()
    source.wl = 1.064
    source.init(params)

    geom = Geom()
    geom.init(params)

    sim = Layers(params, source, geom)
    sim.solve()
    return sim

with open("converge_test02.csv", 'a') as f:
    # f.write("nmodes,R,T,Fx,Fy,Fz,time\n")
    for nmode in range(3,11,2):
        print(f"Computing mode {nmode}")
        start_time = time.time()
        sim = converge(nmode)
        end_time = time.time()
        time_usage = (end_time - start_time) * 1000  # ms
        f.write(f"{nmode}, {sim.Rtot}, {sim.Ttot}, {sim.F[0]}, {sim.F[1]}, {sim.F[2]}, {time_usage}\n")

