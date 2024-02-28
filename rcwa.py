import numpy as np
from typing import Tuple, Union

from Params import *
from utils import *


class WaveVectors:
    def __init__(self, source:Source, geom:Structure, params:RCWAParams):
        self.k_inc = np.sqrt(geom.errf*geom.urrf) * source.inc
        self.dtype = params.dtype
        self.Nmodes = params.Nmodes

        Tx = 2*np.pi / geom.period[0]
        Ty = 2*np.pi / geom.period[1]
        self.kx = self.k_inc[0] - params.modex * Tx / source.k0
        self.ky = self.k_inc[1] - params.modey * Ty / source.k0

        self.kz_0 = self.get_kz()
        self.kz_rf = self.get_kz(geom.errf, geom.urrf)
        self.kz_tm = self.get_kz(geom.ertm, geom.urtm)

    def get_kz(self, er=1.-0j, ur=1.-0j) -> np.ndarray:
        """Kz is always a diagonal matrix, return the diagonal elements"""
        kx, ky = self.kx, self.ky
        if er.imag == 0 and ur.imag == 0:
            n2_conj = er*ur
        else:
            n2_conj = np.conj(er*ur)
        kz = np.conj(np.sqrt(n2_conj - kx*kx - ky*ky))
        return kz

    def homo_decompose(self, er:complex, ur:complex)->Tuple[MAT, MAT, MAT]:
        W = DiagMatrix( np.ones((2, self.Nmodes)) )
        kz = self.get_kz(er, ur)
        Lam = DiagMatrix( np.array([1j*kz, 1j*kz]) )
        Q_data = [self.kx*self.ky, er-self.kx*self.kx, self.ky*self.ky-er, -self.ky*self.kx]
        Q = DiagBlockMatrix(np.array(Q_data))
        V = Q @ Lam.inv()
        return W, Lam, V

    # @log(msg="eigendecomposition")
    def general_decompose(self, er:Union[np.ndarray, complex], ur:Union[np.ndarray, complex])->Tuple[MAT, MAT, MAT]:
        P = self._get_PQ(er, ur)
        Q = self._get_PQ(ur, er)
        omg2 = P @ Q
        lam2, _W = np.linalg.eig(omg2.view())
        W = BlockMatrix(_W)
        lam = np.sqrt(lam2)
        Lam = DiagMatrix(lam)
        V = Q @ W @ Lam.inv()
        return W, Lam, V

    def _get_PQ(self, arr1:Union[np.ndarray, complex], arr2:Union[np.ndarray, complex])->BlockMatrix:
        """arr1 is the component that needs inversion; (er, ur) for P and (ur, er) for Q"""
        kx, ky = self.kx, self.ky
        Kx = np.diag(kx)
        Ky = np.diag(ky)
        if isinstance(arr1, complex):
            d1 = np.diag(kx/arr1*ky)
            d4 = np.diag(-ky/arr1*kx)
            _d2 = kx/arr1*kx
            _d3 = ky/arr1*ky
            if isinstance(arr2, complex):
                d2 = np.diag(arr2 - _d2)
                d3 = np.diag(arr2 - _d3)
            elif isinstance(arr2, np.ndarray):
                d2 = arr2.copy()
                d3 = arr2.copy()
                np.fill_diagonal(d2, d2.diagonal()-_d2)
                np.fill_diagonal(d3, d3.diagonal()-_d3)
                d3 = -1 * d3
        elif isinstance(arr1, np.ndarray):
            kx_arr1_inv = rdiv(arr1, Kx)
            ky_arr1_inv = rdiv(arr1, Ky)
            d1 = kx_arr1_inv * ky
            d4 = -1 * ky_arr1_inv * kx
            _d2 = kx_arr1_inv * kx
            _d3 = ky_arr1_inv * ky
            if isinstance(arr2, complex):
                np.fill_diagonal(_d2, _d2.diagonal()-arr2)
                np.fill_diagonal(_d3, _d3.diagonal()-arr2)
                d2 = -1 * _d2
                d3 = _d3
            elif isinstance(arr2, np.ndarray):
                d2 = arr2 - _d2
                d3 = _d3 - arr2
        return BlockMatrix( np.array([d1, d2, d3, d4]) )




class Layer:
    def __init__(self, er, ur, h=0.):
        self.is_homo = not (isinstance(er, np.ndarray) or isinstance(ur, np.ndarray))
        self.nr = np.sqrt(er*ur)
        self.er = er
        self.ur = ur
        self.h = h
        self.er_fft = None
        self.ur_fft = None

    def init(self, params:RCWAParams, wvm:WaveVectors):
        if self.is_homo:
            self.W, self.Lam, self.V = wvm.homo_decompose(self.er, self.ur)
        else:
            self.er_fft, er = self._init_convol_mat(self.er, params, buffer=self.er_fft)
            self.ur_fft, ur = self._init_convol_mat(self.ur, params, buffer=self.ur_fft)
            self.W, self.Lam, self.V = wvm.general_decompose(er, ur)
        return self.W, self.Lam, self.V

    def _init_convol_mat(self, component, params, buffer=None)->Tuple[Union[np.ndarray, None], Union[np.ndarray, complex]]:
        if isinstance(component, np.ndarray):
            if buffer is None: buffer = fft2(component)
            comp_mn = roll(buffer, params.Mx, params.My)
            conv = convol_matrix(comp_mn, params.Mx, params.My)
        elif isinstance(component, complex):
            conv = component
        return buffer, conv

    # @log(msg="constructing S-matrix")
    def get_Smat(self, params, K, W0, V0, k0):
        W, Lam, V = self.init(params, K)
        if self.is_homo:
            Smat = get_homo_Smatrix(params.Nmodes, Lam, V, V0, k0, self.h)
        else:
            Smat = get_Smatrix(W, Lam, V, W0, V0, k0, self.h)
        return Smat


class Layers(list):
    def __init__(self, params:RCWAParams, src:Source, geom:Structure):
        self.params = params
        self.src = src
        self.geom = geom
        self._init(params, src, geom)

    def _init(self, params, src, geom):
        self.K = WaveVectors(src, geom, params)
        self.e_src = src.e_src

        self.ref_layer = Layer(geom.errf, geom.urrf)
        self.trm_layer = Layer(geom.ertm, geom.urtm)
        self.gap_layer = Layer(1-0j, 1-0j)

        Vrf = self.ref_layer.init(params, self.K)[-1]
        Vtm = self.trm_layer.init(params, self.K)[-1]
        V0 = self.gap_layer.init(params, self.K)[-1]

        self.ref_Smat = get_refSmatrix(params.Nmodes, Vrf, V0)
        self.trm_Smat = get_trmSmatrix(params.Nmodes, Vtm, V0)

        self.nlayers = len(geom.hs)
        self.layers = [Layer(geom.er[i], geom.ur[i], geom.hs[i]) for i in range(self.nlayers)]
        super().__init__(self.layers)
        self.dev_Smat = None
    
    def solve(self):
        n = 0
        Smats = [self.ref_Smat]
        W0 = self.gap_layer.W
        V0 = self.gap_layer.V
        k0 = self.src.k0
        Nmodes = self.params.Nmodes
        for layer in self.layers:
            print(f"Solving for layer {n}...")
            n += 1
            Smat = layer.get_Smat(self.params, self.K, W0, V0, k0)
            Smats.append(Smat)
        Smats.append(self.trm_Smat)
        self.Smat = get_total_Smat(*Smats)
        print("Solving for the fields and diffraction efficiency...")
        self.get_DE()
        self.is_conserve = self._power_conserve()
        self.get_force()

    def get_DE(self):
        # self.e_ref = self.ref_layer.W @ self.Smat.S11 @ inv(self.ref_layer.W) @ self.e_src
        # self.e_trm = self.trm_layer.W @ self.Smat.S21 @ inv(self.ref_layer.W) @ self.e_src
        self.e_ref = self.Smat.S11.view() @ self.e_src
        self.e_trm = self.Smat.S21.view() @ self.e_src
        rx = self.e_ref.T[:self.params.Nmodes]
        ry = self.e_ref.T[self.params.Nmodes:]
        rz = - 1/(self.K.kz_rf) * (self.K.kx*rx + self.K.ky*ry)
        tx = self.e_trm.T[:self.params.Nmodes]
        ty = self.e_trm.T[self.params.Nmodes:]
        tz = - 1/(self.K.kz_tm) * (self.K.kx*tx + self.K.ky*ty)
        self.rcoeff = np.array([rx, ry, rz])
        self.tcoeff = np.array([tx, ty, tz])
        r2 = np.sum(np.abs(self.rcoeff)**2, axis=0)
        t2 = np.sum(np.abs(self.tcoeff)**2, axis=0)
        ck = np.real(self.K.k_inc[2]/self.ref_layer.nr)
        crf = np.real(self.K.kz_rf/self.ref_layer.nr) / ck
        ctm = np.real(self.K.kz_tm/self.trm_layer.nr) / ck
        self.Ref = (crf * r2).reshape(self.params.Nmx, self.params.Nmy)
        self.Trm = (ctm * t2).reshape(self.params.Nmx, self.params.Nmy)
        self.Rtot = np.sum(self.Ref)
        self.Ttot = np.sum(self.Trm)

    def get_force(self, file=''):
        Nmx, Nmy = self.params.Nmx, self.params.Nmy
        Kx = np.real(self.K.kx.reshape(Nmx, Nmy))
        Ky = np.real(self.K.ky.reshape(Nmx, Nmy))
        Kz_rf = self.K.kz_rf.reshape(Nmx, Nmy)
        Kz_tm = self.K.kz_tm.reshape(Nmx, Nmy)
        mask_rf = np.nonzero(np.real(Kz_rf))
        mask_tm = np.nonzero(np.real(Kz_tm))
        crf = np.sqrt(1 - np.real(Kz_rf[mask_rf]/self.ref_layer.nr)**2)
        ctm = np.sqrt(1 - np.real(Kz_tm[mask_tm]/self.trm_layer.nr)**2)
        phi_rf = np.arctan2(Ky[mask_rf], Kx[mask_rf])
        phi_tm = np.arctan2(Ky[mask_tm], Kx[mask_tm])
        Fx = -np.sum(self.Ref[mask_rf]*crf*np.cos(phi_rf) + self.Trm[mask_tm]*ctm*np.cos(phi_tm))
        Fy = -np.sum(self.Ref[mask_rf]*crf*np.sin(phi_rf) + self.Trm[mask_tm]*ctm*np.sin(phi_tm))
        Fz = 1 - np.sum(-self.Ref[mask_rf]*np.real(Kz_rf[mask_rf]/self.ref_layer.nr) + self.Trm[mask_tm]*np.real(Kz_tm[mask_tm]/self.trm_layer.nr))
        self.F = np.array([Fx, Fy, Fz])
        if file:
            np.savez(
                file,
                Kx = np.real(self.K.kx).reshape(self.params.Nmx, self.params.Nmy),
                Ky = np.real(self.K.ky).reshape(self.params.Nmx, self.params.Nmy),
                Kz_rf = self.K.kz_rf.reshape(Nmx, Nmy),
                Kz_tm = self.K.kz_tm.reshape(Nmx, Nmy),
                Kz_0 = self.K.kz_0.reshape(Nmx, Nmy),
                Ref = self.Ref,
                Trm = self.Trm,
                F = self.F,
            )
        return self.F

    def _power_conserve(self):
        return np.isclose(self.Rtot + self.Ttot, 1)
