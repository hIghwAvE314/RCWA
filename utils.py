import numpy as np
from typing import Union
from scipy import linalg as LA

from Params import *

MAT = Union["BlockMatrix", "DiagMatrix", "DiagBlockMatrix"]


class BlockMatrix:
    def __init__(self, data: np.ndarray):
        """data is the whole matrix with the shape (2n, 2n) or (4, n, n), given that n is the size of each block"""
        if len(data.shape) == 2:
            self.view_data = data
            N = data.shape[0]//2
            self.d1 = data[:N, :N]
            self.d2 = data[:N, N:]
            self.d3 = data[N:, :N]
            self.d4 = data[N:, N:]
            self.data = np.array([self.d1, self.d2, self.d3, self.d4])
            assert data.shape[0] == data.shape[1]
            self.block_size = N
        elif len(data.shape) == 3:
            self.data = data
            self.d1, self.d2, self.d3, self.d4 = data
            self.view_data = np.block([[self.d1, self.d2], [self.d3, self.d4]])
            assert data.shape[1] == data.shape[2]
            self.block_size = data.shape[1]
        self.inversion = None

    def inv(self) -> MAT:
        if self.inversion is None:
            self.inversion = BlockMatrix(np.linalg.inv(self.data))
            self.inversion.inversion = self
        return self.inversion

    def view(self) -> np.ndarray:
        return self.view_data

    def __mul__(self, value) -> MAT:
        return BlockMatrix(value * self.data)

    def __rmul__(self, value) -> MAT:
        return BlockMatrix(value * self.data)

    def __add__(self, other: MAT) -> MAT:
        if isinstance(other, DiagMatrix):
            diag = other.diag_data.ravel()
            data = self.view_data.copy()
            np.fill_diagonal(data, data.diagonal()+diag)
            return BlockMatrix(data)
        elif isinstance(other, DiagBlockMatrix):
            data = self.data.copy()
            np.fill_diagonal(data[0], self.d1.diagonal()+other.d1)
            np.fill_diagonal(data[1], self.d2.diagonal()+other.d2)
            np.fill_diagonal(data[2], self.d3.diagonal()+other.d3)
            np.fill_diagonal(data[3], self.d4.diagonal()+other.d4)
            return BlockMatrix(data)
        elif isinstance(other, BlockMatrix):
            return BlockMatrix(self.data + other.data)

    def __sub__(self, other: MAT) -> MAT:
        return self + other*(-1)

    def __matmul__(self, other: MAT) -> MAT:
        if isinstance(other, DiagMatrix):
            diag = other.diag_data.ravel()
            return BlockMatrix(self.view_data * diag)
        elif isinstance(other, DiagBlockMatrix):
            data = np.empty_like(self.data)
            data[0] = self.d1*other.d1 + self.d2*other.d3
            data[1] = self.d1*other.d2 + self.d2*other.d4
            data[2] = self.d3*other.d1 + self.d4*other.d3
            data[3] = self.d3*other.d2 + self.d4*other.d4
            return BlockMatrix(data)
        elif isinstance(other, BlockMatrix):
            return BlockMatrix(self.view_data @ other.view_data)


class DiagBlockMatrix:
    def __init__(self, data: np.ndarray):
        """
        data : np.array([d1, d2, d3, d4]) where d1-4 are numpy 1d array representing the diagonal elements of each block
        the matrix has the form: [[D1, D2], [D3, D4]]
        """
        self.data = data
        self.d1 = data[0]
        self.d2 = data[1]
        self.d3 = data[2]
        self.d4 = data[3]
        self.inversion = None
        self.block_size = self.data.shape[1]

    def inv(self) -> MAT:
        if self.inversion is None:
            det = self.d1*self.d4 - self.d2*self.d3
            inv_d1 = self.d4/det
            inv_d2 = -1 * self.d2/det
            inv_d3 = -1 * self.d3/det
            inv_d4 = self.d1/det
            inv_data = np.array([inv_d1, inv_d2, inv_d3, inv_d4])
            self.inversion = DiagBlockMatrix(inv_data)
            self.inversion.inversion = self
        return self.inversion

    def view(self) -> np.ndarray:
        b1 = np.diag(self.d1)
        b2 = np.diag(self.d2)
        b3 = np.diag(self.d3)
        b4 = np.diag(self.d4)
        return np.block([[b1, b2], [b3, b4]])

    def __mul__(self, value) -> MAT:
        return DiagBlockMatrix(value * self.data)

    def __rmul__(self, value) -> MAT:
        return DiagBlockMatrix(value * self.data)

    def __sub__(self, other: MAT) -> MAT:
        return self + other*(-1)

    def __add__(self, other: MAT) -> MAT:
        if isinstance(other, DiagMatrix):
            data = self.data.copy()
            data[0] += other.d1
            data[3] += other.d4
            return DiagBlockMatrix(data)
        elif isinstance(other, DiagBlockMatrix):
            return DiagBlockMatrix(self.data + other.data)
        elif isinstance(other, BlockMatrix):
            return other + self

    def __matmul__(self, other: MAT) -> MAT:
        if isinstance(other, DiagMatrix):
            d1 = self.d1*other.d1
            d2 = self.d2*other.d4
            d3 = self.d3*other.d1
            d4 = self.d4*other.d4
            data = np.array([d1, d2, d3, d4])
            return DiagBlockMatrix(data)
        elif isinstance(other, DiagBlockMatrix):
            d1 = self.d1*other.d1 + self.d2*other.d3
            d2 = self.d1*other.d2 + self.d2*other.d4
            d3 = self.d3*other.d1 + self.d4*other.d3
            d4 = self.d3*other.d2 + self.d4*other.d4
            data = np.array([d1, d2, d3, d4])
            return DiagBlockMatrix(data)
        elif isinstance(other, BlockMatrix):
            data = np.empty_like(other.data)
            data[0] = (other.d1.T*self.d1 + other.d3.T*self.d2).T
            data[1] = (other.d4.T*self.d2 + other.d2.T*self.d1).T
            data[2] = (other.d1.T*self.d3 + other.d3.T*self.d4).T
            data[3] = (other.d4.T*self.d4 + other.d2.T*self.d3).T
            return BlockMatrix(data)


class DiagMatrix(DiagBlockMatrix):
    def __init__(self, data: np.ndarray):
        if len(data.shape) == 2:
            d1, d4 = data
        elif len(data.shape) == 1:
            N = data.shape[0]//2
            d1 = data[:N]
            d4 = data[N:]
        self.d1 = d1
        self.d4 = d4
        self.d2 = np.zeros_like(d1)
        self.d3 = np.zeros_like(d1)
        self.data = np.array([self.d1, self.d2, self.d3, self.d4])
        self.diag_data = np.array([d1, d4])
        self.inversion = None
        self.block_size = d1.shape[0]

    def inv(self) -> MAT:
        if self.inversion is None:
            self.inversion = DiagMatrix(1/self.diag_data)
            self.inversion.inversion = self
        return self.inversion

    def view(self) -> np.ndarray:
        diag = self.diag_data.ravel()
        return np.diag(diag)

    def __mul__(self, value) -> MAT:
        return DiagMatrix(value * self.diag_data)

    def __rmul__(self, value) -> MAT:
        return DiagMatrix(value * self.diag_data)

    def __sub__(self, other: MAT) -> MAT:
        return self + other*(-1)

    def __add__(self, other: MAT) -> MAT:
        if isinstance(other, DiagMatrix):
            return DiagMatrix(self.diag_data + other.diag_data)
        elif isinstance(other, DiagBlockMatrix):
            return other + self
        elif isinstance(other, BlockMatrix):
            return other + self

    def __matmul__(self, other: MAT) -> MAT:
        if isinstance(other, DiagMatrix):
            d1 = self.d1*other.d1
            d4 = self.d4*other.d4
            data = np.array([d1, d4])
            return DiagMatrix(data)
        elif isinstance(other, DiagBlockMatrix):
            d1 = self.d1*other.d1
            d2 = self.d1*other.d2
            d3 = self.d4*other.d3
            d4 = self.d4*other.d4
            data = np.array([d1, d2, d3, d4])
            return DiagBlockMatrix(data)
        elif isinstance(other, BlockMatrix):
            diag = self.diag_data.ravel()
            data = (other.view_data.T * diag).T
            return BlockMatrix(data)


class SMatrix:
    def __init__(self, S11: MAT, S12: MAT, S21: MAT, S22: MAT):
        assert S11.block_size == S12.block_size == S21.block_size == S22.block_size
        self.S11 = S11
        self.S12 = S12
        self.S21 = S21
        self.S22 = S22
        self.size = self.S11.block_size
        self.is_full = any([isinstance(block, DiagBlockMatrix)
                           for block in (S11, S12, S21, S22)])

    def _load(self):
        return self.S11, self.S12, self.S21, self.S22

    def __mul__(self, other: "SMatrix"):
        assert self.size == other.size
        I = DiagMatrix(np.ones((2, self.size)))
        A11, A12, A21, A22 = self._load()
        B11, B12, B21, B22 = other._load()
        term1 = rdiv(I - B11@A22, A12)
        term2 = rdiv(I - A22@B11, B21)
        C11 = A11 + term1@B11@A21
        C22 = B22 + term2@A22 @ B12
        C12 = term1@B12
        C21 = term2@A21
        return SMatrix(C11, C12, C21, C22)


def div(A: Union[MAT, np.ndarray], B: Union[MAT, np.ndarray]) -> Union[MAT, np.ndarray]:
    """Calculate inv(A)@B"""
    if isinstance(A, DiagBlockMatrix):
        return A.inv() @ B
    elif isinstance(A, BlockMatrix):
        return BlockMatrix(np.linalg.solve(A.view(), B.view()))
    elif isinstance(A, np.ndarray):
        if isinstance(B, np.ndarray):
            return np.linalg.solve(A, B)
        else:
            return np.linalg.solve(A, B.view())


def rdiv(A: Union[MAT, np.ndarray], B: Union[MAT, np.ndarray]) -> Union[MAT, np.ndarray]:
    """Calculate B@inv(A) by using B@inv(A) = solve(A.T, B.T).T if A is BlockMatrix"""
    if isinstance(A, DiagBlockMatrix):
        return B @ A.inv()
    elif isinstance(A, BlockMatrix):
        return BlockMatrix(np.linalg.solve(A.view().T, B.view().T).T)
    elif isinstance(A, np.ndarray):
        if isinstance(B, np.ndarray):
            return np.linalg.solve(A.T, B.T).T
        else:
            return np.linalg.solve(A.T, B.view().T).T


# @log(msg="calculating FFT")
def fft2(arr: np.ndarray) -> np.ndarray:
    Nxy = np.product(arr.shape)  # total number of points in real space
    Arr = np.fft.fft2(arr)/Nxy  # Fourier transform of arr and normlisation
    return Arr


def roll(arr: np.ndarray, Mx: int, My: int) -> np.ndarray:
    """arr: the array to be transformed; Mx, My: range of frequency space (-Mx..Mx) (-My..My)"""
    Arr = np.roll(arr, (Mx, My), axis=(0, 1))[
        :2*Mx+1, :2*My+1]  # truncate the wanted frequencies
    return Arr


def convol_matrix(mat: np.ndarray, Mx: int, My: int) -> np.ndarray:
    Nmodes = (Mx*2+1)*(My*2+1)
    k, l = np.meshgrid(range(Nmodes), range(Nmodes), indexing='ij')
    m, n = np.divmod(k, My*2+1)
    p, q = np.divmod(l, My*2+1)
    idx = np.rint(m-p + Mx).astype(int)
    idy = np.rint(n-q + My).astype(int)
    cond = ((0 <= idx)*(idx < Mx*2+1)) * ((0 <= idy)*(idy < My*2+1))
    idx = np.where(cond, idx, 0)
    idy = np.where(cond, idy, 0)
    return np.where(cond, mat[idx, idy], 0)


def get_homo_Smatrix(Nmodes: int, Lam: DiagMatrix, V: DiagBlockMatrix, V0: DiagBlockMatrix, k0: float, thick: float = 0.) -> SMatrix:
    Wtot = DiagMatrix(np.ones((2, Nmodes)))
    Vtot = V.inv() @ V0
    A = Wtot + Vtot
    B = Wtot - Vtot
    X = DiagMatrix(np.exp(-k0*thick * Lam.diag_data))
    BA_inv = B @ A.inv()
    D = X@BA_inv@X
    F = (A - D@B).inv()
    S11 = F @ (D@A - B)
    S12 = F @ X @ (A - BA_inv@B)
    S21 = S12
    S22 = S11
    return SMatrix(S11, S12, S21, S22)


def get_Smatrix(W: MAT, Lam: DiagMatrix, V: MAT, W0: DiagMatrix, V0: DiagBlockMatrix, k0: float, thick: float = 0.) -> SMatrix:
    Wtot = div(W, W0)
    Vtot = div(V, V0)
    A = Wtot + Vtot
    B = Wtot - Vtot
    X = DiagMatrix(np.exp(-k0*thick * Lam.diag_data))
    BA_inv = rdiv(A, B)
    D = X@BA_inv@X
    F_lu = LA.lu_factor((A - D@B).view())
    S11 = BlockMatrix(LA.lu_solve(F_lu, (D@A-B).view()))
    S12 = BlockMatrix(LA.lu_solve(F_lu, (X@(A - BA_inv@B)).view()))
    S21 = S12
    S22 = S11
    return SMatrix(S11, S12, S21, S22)


def get_refSmatrix(Nmodes: int, V: DiagBlockMatrix, V0: DiagBlockMatrix) -> SMatrix:
    Wtot = DiagMatrix(np.ones((2, Nmodes)))
    Vtot = V0.inv() @ V
    A = Wtot + Vtot
    B = Wtot - Vtot
    BA_inv = B@A.inv()
    S11 = -1 * A.inv() @ B
    S12 = 2*A.inv()
    S21 = 0.5 * (A - BA_inv@B)
    S22 = BA_inv
    return SMatrix(S11, S12, S21, S22)


def get_trmSmatrix(Nmodes: int, V: DiagBlockMatrix, V0: DiagBlockMatrix) -> SMatrix:
    Wtot = DiagMatrix(np.ones((2, Nmodes)))
    Vtot = V0.inv() @ V
    A = Wtot + Vtot
    B = Wtot - Vtot
    BA_inv = B@A.inv()
    S22 = -1 * A.inv() @ B
    S21 = 2*A.inv()
    S12 = 0.5 * (A - BA_inv@B)
    S11 = BA_inv
    return SMatrix(S11, S12, S21, S22)


def get_total_Smat(*Smats: SMatrix) -> SMatrix:
    print("Computing total S-matrix")
    new_Smats = []
    last = None
    for n, Smat in enumerate(Smats):
        if last is None:
            last = Smat
            if Smat.is_full:
                new_Smats.append(Smat)
                last = None
        else:
            if not Smat.is_full:
                last = last * Smat
            else:
                new_Smats.append(last)
                new_Smats.append(Smat)
                last = None
    if last is not None:
        new_Smats.append(last)
    tot = None
    for Smat in new_Smats:
        tot = tot * Smat if tot is not None else Smat
    return tot
