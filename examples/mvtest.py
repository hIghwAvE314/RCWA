import sys, os
sys.path.append('..')

from rcwa import RCWAParams, Source, Layers
from geometries import MetaVehicle

params = RCWAParams()
params.Nmx, params.Nmy = 35, 11
params.init()

source = Source()
source.wl = 1.064
source.init(params)

geom = MetaVehicle()
geom.init(params)

sim = Layers(params, source, geom)
sim.solve()
print(sim.F)