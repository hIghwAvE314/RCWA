from typing import Union
import numpy as np
import scipy as sp
import abc
from scipy import linalg as LA
from scipy import sparse as spa
from scipy.sparse import linalg as spLA
import matplotlib.pyplot as plt
from collections import namedtuple

from Params import *


MAT = Union[np.ndarray, spa.csc_matrix]
# Smatrix = namedtuple('Smatrix', ['S11', 'S12', 'S21', 'S22'])


class WaveVectorMatrix:
    def __init__(self, source:Source, geom:Structure, params:RCWAParams):
        self.k_inc = np.sqrt(geom.errf*geom.urrf) * source.inc
        Tx = 2*np.pi / geom.period[0]
        Ty = 2*np.pi / geom.period[1]
        kx = self.k_inc[0] - params.modex * Tx / source.k0
        ky = self.k_inc[1] - params.modey * Ty / source.k0

        self.Kx = spa.diags(kx, format='csc', dtype=complex)
        self.Ky = spa.diags(ky, format='csc', dtype=complex)
        self.Kz_0 = get_Kz(self.Kx, self.Ky)
        self.Kz_rf = get_Kz(self.Kx, self.Ky, geom.errf, geom.urrf)
        self.Kz_tm = get_Kz(self.Kx, self.Ky, geom.ertm, geom.urtm)


class Smatrix:
    def __init__(self, S11, S12, S21, S22):
        self.S11 = S11
        self.S12 = S12
        self.S21 = S21
        self.S22 = S22
        self.Smat = block([[S11, S12], [S21, S22]])
    
    def __mul__(self, other):
        return redheffer_product(self, other)

    def det(self):
        return LA.det(self.Smat)


def div(a, b):
    return np.divide(a, b, out=np.zeros_like(b), where=b!=0)

def block_diag_inv(A:spa.spmatrix):
    M = int(A.shape[0]/2)
    diag0 = A.diagonal(k=0)
    diag1 = A.diagonal(k=M)
    diag_1 = A.diagonal(k=-M)
    a = diag0[:M]
    b = diag1
    c = diag_1
    d = diag0[M:]
    t1 = div(1,(a - div(b,d)*c))
    t2 = div(1,(d - div(c,a)*b))
    a11 = spa.diags(t1, format='csc', dtype=complex)
    a12 = spa.diags(-t1*div(b,d), format='csc', dtype=complex)
    a21 = spa.diags(-t2*div(c,a), format='csc', dtype=complex)
    a22 = spa.diags(t2, format='csc', dtype=complex)
    return spa.bmat([[a11, a12], [a21, a22]], format='csc', dtype=complex)


def is_diag(A:MAT):
    nz = A.count_nonzero() if isinstance(A, spa.spmatrix) else np.count_nonzero(A)
    dnz = np.count_nonzero(A.diagonal())
    return nz == dnz

def is_sparse(A:MAT):
    nnz = A.nnz if isinstance(A, spa.spmatrix) else np.count_nonzero(A)
    sparsity = nnz/np.product(A.shape)
    return sparsity <= 0.25

def sparsity(A:MAT) -> float:
    nnz = A.nnz if isinstance(A, spa.spmatrix) else np.count_nonzero(A)
    sparsity = nnz/np.product(A.shape)
    return sparsity

def det(A:MAT) -> float:
    if isinstance(A, spa.spmatrix):
        lu = spLA.splu(A)
        diagL = lu.L.diagonal()
        diagU = lu.U.diagonal()
        return diagL.prod() * diagU.prod()
    return LA.det(A)


def dia_inverse(A:MAT) -> spa.spmatrix:
    diag = A.diagonal() if isinstance(A, spa.spmatrix) else np.diag(A)
    diag_inv = np.divide(1, diag, out=np.zeros_like(diag, dtype=complex), where=diag!=0)
    inv = spa.diags(diag_inv, format='csc', dtype=complex)
    return inv

def diag_expm(A:spa.spmatrix, acc=0):
    """calculate the exponential matrix expm(kA) for a diagnol matrix A, k is the coefficient"""
    diag = A.diagonal() if isinstance(A, spa.spmatrix) else np.diag(A)
    res = spa.diags(np.exp(diag), format='csc', dtype=complex)
    return reduce(res, acc=acc)

def reduce(A:MAT, acc=0):
    """Return sparse matrix depending on the sparsity, every value is round to the given decimal places"""
    if acc:
        if isinstance(A, spa.spmatrix): 
            if is_sparse(A): 
                res = np.round(A, acc)
                res.eliminate_zeros()
                return res
            else:
                A = A.toarray()
        A_red = np.round(A, acc)
        if is_sparse(A_red):
            return spa.csc_matrix(A_red, dtype=complex)
        return A_red
    return A

@timer("Calculating the matrix inversion")
def pinv(A:MAT, is_homo=False, acc=0) -> MAT:
    """return the inverse of A with same type of A"""
    if is_homo or is_diag(A):
        return dia_inverse(A)
    if isinstance(A, spa.spmatrix):
        if is_sparse(A):
            try:
                res = spLA.inv(A)
            except RuntimeError:
                A = A.toarray()
                res = LA.pinv(A)
        else:
            A = A.to_array()
            res = LA.pinv(A)
    else:
        res = LA.pinv(A)
    return reduce(res, acc=acc)

@timer("Calculating LU solve")
def divide(A:MAT, B:MAT, is_homo=False, acc=0) -> MAT:
    """Calculate inv(A) @ B by solving linear system (LU solve or spsolve)"""
    if is_homo or is_diag(A):
        return dia_inverse(A) @ B
    if isinstance(A, spa.spmatrix) and is_sparse(A):
        try:
            res = spLA.spsolve(A, B)
        except RuntimeError:
            res = pinv(A.toarray(), acc=acc) @ B
    else:
        if isinstance(A, spa.spmatrix): A = A.toarray()
        try:
            if isinstance(B, spa.spmatrix): B = B.toarray()
            res = LA.lu_solve(LA.lu_factor(A), B)
        except RuntimeError:
            res = reduce(pinv(A, acc=acc) @ B, acc=acc)
    return reduce(res, acc=acc)

@timer("Calculating the redheffer star product")
def redheffer_product(SA:Smatrix, SB:Smatrix, acc=0) -> Smatrix:
    """Calculate the redheffer star product of matrix SA and SB, each component of Smatrix should be MAT type"""
    I = spa.identity(SA.S11.shape[0], format='csc', dtype=complex)
    mat1 = reduce(I - SB.S11@SA.S22, acc=acc)
    mat2 = reduce(I - SA.S22@SB.S11, acc=acc)
    term1 = reduce(SA.S12 @ pinv(mat1, acc=acc), acc=acc)
    term2 = reduce(SB.S21 @ pinv(mat2, acc=acc), acc=acc)
    
    C11 = reduce(SA.S11 + term1 @ SB.S11@SA.S21, acc=acc)
    C12 = reduce(term1 @ SB.S12, acc=acc)
    C21 = reduce(term2 @ SA.S21, acc=acc)
    C22 = reduce(SB.S22 + term2 @ SA.S22 @ SB.S12, acc=acc)
    C = Smatrix(C11, C12, C21, C22)
    return C

@timer("Calculating FFT of the sturcture")
def fft2(arr:np.ndarray) -> np.ndarray:
    Nxy = np.product(arr.shape)  # total number of points in real space
    Arr = np.fft.fft2(arr)/Nxy  # Fourier tranform of arr (normalised)
    return Arr

@timer("Truncating the Fourier Series of the sturcture")
def roll(arr:np.ndarray, Mx:int, My:int) -> np.ndarray:
    """arr: the array to be transformed; Mx, My: range of frequency space (-Mx..Mx) (-My..My)"""
    Arr = np.roll(arr, (Mx, My), axis=(0,1))[:2*Mx+1, :2*My+1]  # truncate the wanted frequencies
    return Arr

@timer("Constructing the convolution matrix")
def convol_matrix(mat:np.ndarray, Mx:int, My:int) -> np.ndarray:
    Nmodes = (Mx*2+1)*(My*2+1)
    k, l = np.meshgrid(range(Nmodes), range(Nmodes), indexing='ij')
    m, n = np.divmod(k, My*2+1)
    p, q = np.divmod(l, My*2+1)
    idx = np.rint(m-p + Mx).astype(int)
    idy = np.rint(n-q + My).astype(int)
    cond = ((0<=idx)*(idx<Mx*2+1)) * ((0<=idy)*(idy<My*2+1))
    idx = np.where(cond, idx, 0)
    idy = np.where(cond, idy, 0)
    return np.where(cond, mat[idx, idy], 0) 

def block(blocks):
    for i, row in enumerate(blocks):
        for j, item in enumerate(row):
            if isinstance(item, spa.spmatrix):
                blocks[i][j] = item.toarray()
    return np.block(blocks)

def get_Kz(Kx:spa.csc_matrix, Ky:spa.csc_matrix, er=1.+0j, ur=1.+0j) -> spa.spmatrix:
    """Kz is always a diagonal matrix"""
    I = spa.identity(Kx.shape[0], dtype=complex, format='csc')
    return np.conj(np.sqrt(np.conj(er*ur)*I - Kx@Kx - Ky@Ky))

def homo_Qmatrix(Kx:spa.csc_matrix, Ky:spa.csc_matrix, er=1.+0j, ur=1.+0j) -> spa.spmatrix:
    """homoQmatrix is always a diagonal matrix"""
    I = spa.identity(Kx.shape[0], dtype=complex, format='csc')
    Q = [[Kx@Ky, ur*er*I - Kx@Kx], [Ky@Ky - ur*er*I, -Ky@Kx]]
    return 1/ur * spa.bmat(Q, format='csc', dtype=complex)

@timer("Generating Q matrix for general layer")
def general_Qmatrix(Kx:spa.csc_matrix, Ky:spa.csc_matrix, er:np.ndarray, ur:np.ndarray, acc=0) -> spa.spmatrix:
    """Usually general Q matrix is block-wise half sparse matrix, thus we treat Q as a dense matrix"""
    if isinstance(ur, np.ndarray):
        lu_ur = LA.lu_factor(ur)
        urKy = LA.lu_solve(lu_ur, Ky.toarray())
        urKx = LA.lu_solve(lu_ur, Kx.toarray())
    else:
        urKy = divide(ur, Ky)
        urKx = divide(ur, Kx)
    Q = [[Kx@urKy, er - Kx@urKx], [Ky@urKy - er, -Ky@urKx]]
    return reduce(block(Q), acc=acc)

@timer("Generating P matrix for general layer")
def general_Pmatrix(Kx:spa.csc_matrix, Ky:spa.csc_matrix, er:np.ndarray, ur:np.ndarray, acc=0) -> spa.spmatrix:
    """Usually general P matrix is a dense matrix"""
    if isinstance(er, np.ndarray):
        lu_er = LA.lu_factor(er)
        erKy = LA.lu_solve(lu_er, Ky.toarray())
        erKx = LA.lu_solve(lu_er, Kx.toarray())
    else:
        erKy = divide(er, Ky)
        erKx = divide(er, Kx)
    P = [[Kx@erKy, ur - Kx@erKx], [Ky@erKy - ur, -Ky@erKx]]
    return reduce(block(P), acc=acc)

def homo_decompose(Kx:spa.csc_matrix, Ky:spa.csc_matrix, er=1.+0j, ur=1.+0j):
    """homo W Lam V are all diagonal matrices"""
    W = spa.identity(2*Kx.shape[0], dtype=complex, format='csc')
    Kz = get_Kz(Kx, Ky, er, ur)
    Lam = spa.bmat([[1j*Kz, None], [None, 1j*Kz]], format='csc')
    Q = homo_Qmatrix(Kx, Ky, er, ur)
    V = Q @ dia_inverse(Lam)
    return W, Lam, V

@timer("Eigenvalues decomposition")
def general_decompose(Kx:spa.csc_matrix, Ky:spa.csc_matrix, er:np.ndarray, ur:np.ndarray, acc=0):
    """W and V are dense matrix, Lam is sparse diagonal matrix"""
    P = general_Pmatrix(Kx, Ky, er, ur)
    Q = general_Qmatrix(Kx, Ky, er, ur)
    """ This line below is very expensive """
    omg2 = reduce(P@Q, acc=acc)
    if isinstance(omg2, spa.spmatrix): omg2 = omg2.toarray()
    lam2, W = LA.eig(omg2)  # P,Q here must be dense matrix as it is not possible to calculate full eigenvectors of a sparse matrix?
    lam = np.sqrt(lam2)
    Lam = spa.diags(lam, format='csc', dtype=complex)
    Lam_inv = dia_inverse(Lam)
    V = Q @ W @ Lam_inv
    return W, Lam, V

@timer("Generating the S-matrix of the layer")
def get_Smatrix(W:MAT, Lam:spa.csc_matrix, V:MAT, W0:spa.spmatrix, V0:MAT, k0, thick=0., is_homo=False, acc=0):
    """if is_homo: all components of S is diagonal sparse matrix, else: all component is dense matrix"""
    Wterm = divide(W, W0, is_homo=is_homo)
    Vterm = divide(V, V0)
    A = reduce(Wterm + Vterm, acc=acc)  # MAT
    B = reduce(Wterm - Vterm, acc=acc)  # MAT
    X = diag_expm( -k0*thick * Lam)  # sparse
    BA_inv = reduce(B @ pinv(A), acc=acc)  # MAT
    D = pinv(A - X @ BA_inv @ X @ B, acc=acc)  # MAT
    S11 = reduce(D @ ( X@BA_inv@X@A - B))  # MAT
    S12 = reduce(D @ X @ ( A - BA_inv@B ))  # MAT
    S21 = S12  # MAT
    S22 = S11  # MAT
    return Smatrix(S11, S12, S21, S22)

def get_gapSmatrix(Nmodes:int) -> Smatrix:
    I = spa.identity(2*Nmodes, dtype=complex, format='csc')
    O = spa.csc_matrix((2*Nmodes, 2*Nmodes), dtype=complex)
    return Smatrix(O, I, I, O)

@timer("Generating the S-matrix of reflection side")
def get_refSmatrix(W:spa.spmatrix, V:MAT, W0:spa.spmatrix, V0:MAT, acc=0) -> Smatrix:
    Wterm = reduce(pinv(W0) @ W, acc=acc)  # sparse
    Vterm = reduce(pinv(V0) @ V, acc=acc)  # sparse
    A = reduce(Wterm + Vterm, acc=acc)  # sparse
    B = reduce(Wterm - Vterm, acc=acc)  # sparse
    A_inv = reduce(pinv(A))  # sparse
    S11 = reduce(-A_inv @ B)  # sparse
    S12 = reduce(2 * A_inv)  # sparse
    S21 = reduce(0.5 * ( A - B@A_inv@B ))  # 0.5*(A-B@A_inv@B)  # sparse
    S22 = reduce(B @ A_inv)  # sparse
    return Smatrix(S11, S12, S21, S22)

@timer("Generating the S-matrix of transmission side")
def get_trmSmatrix(W:spa.spmatrix, V:MAT, W0:spa.spmatrix, V0:MAT, acc=0) -> Smatrix:
    Wterm = reduce(pinv(W0) @ W, acc=acc)  # sparse
    Vterm = reduce(pinv(V0) @ V, acc=acc)  # sparse
    A = reduce(Wterm + Vterm, acc=acc)  # sparse
    B = reduce(Wterm - Vterm, acc=acc)  # sparse
    A_inv = reduce(pinv(A))  # sparse
    S22 = reduce(-A_inv @ B)  # sparse
    S21 = reduce(2 * A_inv)  # sparse
    S12 = reduce(0.5 * ( A - B@A_inv@B ))  # sparse
    S11 = reduce(B @ A_inv)  # sparse
    return Smatrix(S11, S12, S21, S22)

@timer("Calculating the total S-matrix")
def get_total_Smat(*Smats, inhomo=[]):
    new_Smats = []
    last = None
    print("Summerizing the sparse S-matrices")
    for n, Smat in enumerate(Smats):
        if n in inhomo:
            if last is not None:
                new_Smats.append(last)
                last = None
            new_Smats.append(Smat)
        else:
            last = last * Smat if last is not None else Smat
    if last is not None:
        new_Smats.append(last)
    tot = None
    print("Summerizing all S-matrices")
    for Smat in new_Smats:
        tot = tot * Smat if tot is not None else Smat
    return tot


    
