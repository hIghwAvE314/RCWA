import numpy as np
import scipy as sp
import abc
from scipy import linalg as LA
from scipy import sparse as spa
from scipy.sparse import linalg as spLA
import matplotlib.pyplot as plt


from Params import *
from matrices import *



class Layer:
    def __init__(self, er, ur, h=0.):
        self.is_homo = not (isinstance(er, np.ndarray) or isinstance(ur, np.ndarray))
        self.nr = np.sqrt(er*ur)
        self.er = er
        self.ur = ur
        self.h = h
        self.er_fft = None
        self.ur_fft = None

    def init(self, params:RCWAParams, K:WaveVectorMatrix):
        if self.is_homo:
            self.W, self.Lam, self.V = homo_decompose(K.Kx, K.Ky, self.er, self.ur)
        else:
            self.er_fft, er = self._init_convol_mat(self.er, params, Comp=self.er_fft)
            self.ur_fft, ur = self._init_convol_mat(self.ur, params, Comp=self.ur_fft)
            self.W, self.Lam, self.V = general_decompose(K.Kx, K.Ky, er, ur, acc=params.acc)
        return self.W, self.Lam, self.V

    def _init_convol_mat(self, comp, params, Comp=None):
        if isinstance(comp, np.ndarray):
            if Comp is None: Comp = fft2(comp)
            comp_mn = roll(Comp, params.Mx, params.My)
            conv = convol_matrix(comp_mn, params.Mx, params.My)
        else:
            conv = comp * spa.identity(params.Nmodes, dtype=complex, format='csc')
        return Comp, conv




class Layers(list):
    def __init__(self, params:RCWAParams, src:Source, geom:Structure):
        self.params = params
        self.src = src
        self.geom = geom
        self._init(params, src, geom)

    @timer("Initialising the layers")
    def _init(self, params, src, geom):
        self.K = WaveVectorMatrix(src, geom, params)
        self.e_src = src.e_src

        self.ref_layer = Layer(geom.errf, geom.urrf)
        self.trm_layer = Layer(geom.ertm, geom.urtm)
        self.gap_layer = Layer(1+0j, 1+0j)

        self.ref_layer.init(params, self.K)
        self.trm_layer.init(params, self.K)
        self.gap_layer.init(params, self.K)
        ref = self.ref_layer
        gap = self.gap_layer
        trm = self.trm_layer

        self.gap_Smat = get_gapSmatrix(params.Nmodes)
        print("Set up the reflection side S-matrix")
        self.ref_Smat = get_refSmatrix(ref.W, ref.V, gap.W, gap.V, acc=params.acc)
        print("Set up the transmission side S-matrix")
        self.trm_Smat = get_trmSmatrix(trm.W, trm.V, gap.W, gap.V, acc=params.acc)

        self.nlayers = len(geom.hs)
        self.layers = [Layer(geom.er[i], geom.ur[i], geom.hs[i]) for i in range(self.nlayers)]
        super().__init__(self.layers)
        self.dev_Smat = self.gap_Smat
    
    @timer("Solving the problems")
    def solve(self):
        n = 0
        inhomo_indices = []
        Smats = [self.ref_Smat]
        for layer in self.layers:
            print(f"Solving for layer {n}...")
            n += 1
            W, Lam, V = layer.init(self.params, self.K)
            Smat = get_Smatrix(W, Lam, V, self.gap_layer.W, self.gap_layer.V, self.src.k0, layer.h, is_homo=layer.is_homo, acc=self.params.acc)
            Smats.append(Smat)
            inhomo_indices.append(n)
            # self.dev_Smat = redheffer_product(self.dev_Smat, Smat, acc=self.params.acc)
        Smats.append(self.trm_Smat)
        # Smat = redheffer_product(self.ref_Smat, self.dev_Smat, acc=self.params.acc)
        # self.Smat = redheffer_product(Smat, self.trm_Smat, acc=self.params.acc)
        self.Smat = get_total_Smat(*Smats)
        print("Solving for the fields and diffraction efficiency...")
        self.e_ref = self.ref_layer.W @ self.Smat.S11 @ dia_inverse(self.ref_layer.W) @ self.e_src
        self.e_trm = self.trm_layer.W @ self.Smat.S21 @ dia_inverse(self.ref_layer.W) @ self.e_src
        rx = self.e_ref.T[:self.params.Nmodes]
        ry = self.e_ref.T[self.params.Nmodes:]
        rz = - dia_inverse(self.K.Kz_rf) @ (self.K.Kx@rx + self.K.Ky@ry)
        tx = self.e_trm.T[:self.params.Nmodes]
        ty = self.e_trm.T[self.params.Nmodes:]
        tz = - dia_inverse(self.K.Kz_tm) @ (self.K.Kx@tx + self.K.Ky@ty)
        self.rcoeff = np.array([rx, ry, rz])
        self.tcoeff = np.array([tx, ty, tz])
        r2 = np.sum(np.abs(self.rcoeff)**2, axis=0)
        t2 = np.sum(np.abs(self.tcoeff)**2, axis=0)
        ck = np.real(self.K.k_inc[2]/self.ref_layer.nr)
        crf = np.real(self.K.Kz_rf/self.ref_layer.nr) / ck
        ctm = np.real(self.K.Kz_tm/self.trm_layer.nr) / ck
        self.Ref = (crf @ r2).reshape(self.params.Nmx, self.params.Nmy)
        self.Trm = (ctm @ t2).reshape(self.params.Nmx, self.params.Nmy)
        self.Rtot = np.sum(self.Ref)
        self.Ttot = np.sum(self.Trm)

        self.is_conserve = self._power_conserve()

    def _power_conserve(self):
        return np.isclose(self.Rtot + self.Ttot, 1)

    def change_nmodes(self, Nmx, Nmy):
        print(f"Changing total modes number to {Nmx},{Nmy}")
        self.params.Nmx = Nmx
        self.params.Nmy = Nmy
        self.params.init()
        self.src.init(self.params)
        self.geom.init(self.params)
        self._init(self.params, self.src, self.geom)
        self.solve()

    def converge_test(self, Max, step=2, comp='xy', acc=1e-6):
        self.solve()
        R = [self.Rtot]
        T = [self.Ttot]
        Mx = self.params.Nmx
        My = self.params.Nmy
        Nx = (Max - Mx) / step 
        Ny = (Max - My) / step 
        N = max(Nx, 1) if 'x' in comp else 1
        N = max(N, Ny) if 'y' in comp else N
        is_converge=False
        for n in range(int(N)):
            Mx = Mx + step if 'x' in comp else Mx
            My = My + step if 'y' in comp else My
            print(Mx, My)
            self.change_nmodes(Mx, My)
            if np.isclose(self.Rtot, R[-1], atol=acc) and np.isclose(self.Ttot, T[-1], atol=acc):
                print(f"Convergence is reached at mode ({Mx},{My}), Reflectance {np.real(self.Rtot)}, Transmittance {np.real(self.Ttot)}")
                is_converge = True
            else:
                is_converge = False
            R.append(self.Rtot)
            T.append(self.Ttot)
        return R, T




