from numbers import Number
import numpy as np
import scipy as sp
from scipy import linalg as LA
from scipy import sparse as spa
from scipy.sparse import linalg as spLA
import torch
from typing import Union
from functools import wraps

from Params import *


MAT = Union[spa.spmatrix, torch.Tensor]

def load_tensor(mat:torch.Tensor):
    """Load tensor from cpu to cuda for GPU computation"""
    return mat.cuda('cuda')

def store_tensor(mat:torch.Tensor):
    """Save tensor from cuda to cpu to free GPU memory"""
    return mat.detach().cpu()

def is_sparse(A:MAT) -> bool:
    return isinstance(A, spa.spmatrix)

def get_diag(A:MAT) -> Union[np.ndarray, None]:
    """Only return the diagonal if the matrix is a sparse diagonal matrix"""
    if hasattr(A, 'diag'):
        return getattr(A, 'diag')
    if is_sparse(A):
        diag = A.diagonal()
        if A.count_nonzero() == np.count_nonzero(diag):
            setattr(A, 'diag', diag)
            return diag
    return None

def totorch(A:MAT, device='cuda') -> torch.Tensor:
    if is_sparse(A):
        return torch.tensor(A.toarray(), device=device)
    if device == 'cpu':
        return A.cpu()
    return A.cuda(device)

def torch2numpy(A:torch.Tensor) -> np.ndarray:
    return A.detach().cpu().numpy()

def numpy2torch(A:np.ndarray, device='cuda') -> torch.Tensor:
    res = torch.from_numpy(A)
    if device == 'cpu':
        return res
    return res.cuda(device)

def divide(num, arr:np.ndarray) -> np.ndarray:
    return np.divide(num, arr, out=np.zeros_like(arr), where=arr!=0)

"""These functions operates different matrix type"""
def matmul(A:MAT, B:MAT) -> MAT:
    if is_sparse(A) and is_sparse(B):
        return A @ B
    return totorch(A) @ totorch(B)

def add(A:MAT, B:MAT) -> MAT:
    if is_sparse(A) and is_sparse(B):
        return A + B
    return totorch(A) + totorch(B)

def sub(A:MAT, B:MAT) -> MAT:
    if is_sparse(A) and is_sparse(B):
        return A - B
    return totorch(A) - totorch(B)

def mul(A:MAT, B:MAT) -> MAT:
    if is_sparse(A) and is_sparse(B):
        return A * B
    return totorch(A) * totorch(B)

def inv(A:MAT, is_store=False) -> MAT:
    if hasattr(A, 'invmat'):
        return getattr(A, 'invmat')
    try:
        if is_sparse(A):
            diag = get_diag(A)
            if diag is None:
                mat = spLA.inv(A)
            else:
                diag_inv = divide(1., diag)
                mat = spa.diags(diag_inv, format='csc', dtype=diag.dtype)
                setattr(mat, 'diag', diag_inv)
        else:
            mat = torch.linalg.inv(A)
    except RuntimeError as E:
        print(E)
        mat = pinv(A)
    if is_store:
        setattr(A, 'invmat', mat)
    return mat

def pinv(A:MAT) -> torch.Tensor:
    """This method is quite slow, don't use unless neccessary"""
    return torch.linalg.pinv(totorch(A))

def expm(A:spa.spmatrix) -> spa.spmatrix:
    """Only exponential of a diagonal sparse matrix is supported"""
    diag = get_diag(A)
    if diag is not None:
        exp_diag = np.exp(diag)
        mat = spa.diags(exp_diag, format='csc', dtype=diag.dtype)
        setattr(mat, 'diag', exp_diag)
        return mat
    raise TypeError("Matrix exponential for a non diagonal sparse matrix is not supported!")

@log("Eigen core")
def eig(A:torch.Tensor, pure_torch=False):
    if is_sparse(A):
        raise TypeError("Sparse matrix full eig is not supported!")
    if pure_torch:
        vals, vecs = torch.linalg.eig(A)
    else:
        Anp = torch2numpy(A)
        npvals, npvecs = LA.eig(Anp)
        # vals = numpy2torch(npvals)
        vals = npvals
        vecs = numpy2torch(npvecs)
    return vals, vecs

def div(A:MAT, B:MAT) -> MAT:
    """ Calculate A_inverse @ B """
    if is_sparse(A):
        diag = get_diag(A)
        if (diag is not None) or (not is_sparse(B)):
            return matmul(inv(A), B)
        A_lu = A.tocsc()
        if is_sparse(B):
            B_lu = B.tocsc()
        return spLA.spsolve(A_lu, B_lu)
    return torch.linalg.solve(A, totorch(B))

def rdiv(A:MAT, B:MAT) -> MAT:
    """Calculate B @ A_inverse by solve(A.T, B.T).T"""
    return div(A.T, B.T).T

@log()
def block(blocks):
    """Be careful of memory usage when using this function"""
    [[A11, A12], [A21, A22]] = blocks
    if all(is_sparse(A) for A in [A11, A12, A21, A22]):
        return spa.bmat(blocks, format='csc')
    row1 = torch.cat([totorch(A11,device='cpu'), totorch(A12,device='cpu')], 1)
    row2 = torch.cat([totorch(A21,device='cpu'), totorch(A22,device='cpu')], 1)
    bmat = torch.cat([row1, row2], 0)
    return bmat.cuda()


class SMatrix:
    @log()
    def __init__(self, S11:MAT, S12:MAT, S21:MAT, S22:MAT):
        self.is_sparse = all(is_sparse(S) for S in [S11, S12, S21, S22])
        self.S11 = totorch(S11, device='cpu') if not self.is_sparse else S11
        self.S12 = totorch(S12, device='cpu') if not self.is_sparse else S12
        self.S21 = totorch(S21, device='cpu') if not self.is_sparse else S21
        self.S22 = totorch(S22, device='cpu') if not self.is_sparse else S22
        self.dtype = self.S11.dtype
        self.size = self.S11.shape[0]

    def _load(self):
        if not self.is_sparse:
            S11 = totorch(self.S11, device='cuda')
            S12 = totorch(self.S12, device='cuda')
            S21 = totorch(self.S21, device='cuda')
            S22 = totorch(self.S22, device='cuda')
            return S11, S12, S21, S22
        return self.S11, self.S12, self.S21, self.S22
    
    @log("Computing S-Matrix star product")
    def __mul__(self, other):
        if self.is_sparse and other.is_sparse:
            I = spa.identity(self.size, format='csc', dtype=self.dtype)
        else:
            if self.is_sparse:
                I = numpy2torch(np.eye(self.size, dtype=self.dtype))
            else:
                I = torch.eye(self.size, device='cuda', dtype=self.dtype)
        A11, A12, A21, A22 = self._load()
        B11, B12, B21, B22 = other._load()
        term1 = rdiv(I - matmul(B11, A22), A12)
        term2 = rdiv(I - matmul(A22, B11), B21)
        C11 = add(A11, matmul(matmul(term1, B11), A21))
        C22 = add(B22, matmul(matmul(term2, A22), B12))
        C12 = matmul(term1, B12)
        C21 = matmul(term2, A21)
        return SMatrix(C11, C12, C21, C22)
        

class WaveVectorMatrix:
    def __init__(self, source:Source, geom:Structure, params:RCWAParams):
        self.k_inc = np.sqrt(geom.errf*geom.urrf) * source.inc
        self.dtype = params.dtype
        self.Nmodes = params.Nmodes

        Tx = 2*np.pi / geom.period[0]
        Ty = 2*np.pi / geom.period[1]
        kx = self.k_inc[0] - params.modex * Tx / source.k0
        ky = self.k_inc[1] - params.modey * Ty / source.k0

        self.Kx = spa.diags(kx, format='csc', dtype=self.dtype)
        self.Ky = spa.diags(ky, format='csc', dtype=self.dtype)
        setattr(self.Kx, 'diag', kx.astype(self.dtype))
        setattr(self.Ky, 'diag', ky.astype(self.dtype))
        self.Kz_0 = self.get_Kz()
        self.Kz_rf = self.get_Kz(geom.errf, geom.urrf)
        self.Kz_tm = self.get_Kz(geom.ertm, geom.urtm)
    
    def get_eye(self, n=1):
        return spa.identity(n*self.Nmodes, dtype=self.dtype, format='csc')

    def get_Kz(self, er=1.+0j, ur=1.+0j) -> spa.spmatrix:
        """Kz is always a diagonal matrix"""
        Kx, Ky, I = self.Kx, self.Ky, self.get_eye()
        Kz = np.conj(np.sqrt(np.conj(er*ur)*I - Kx@Kx - Ky@Ky))
        setattr(Kz, 'diag', Kz.diagonal())
        return Kz

    def _homo_Qmatrix(self, er=1.+0j, ur=1.+0j) -> spa.spmatrix:
        """homoQmatrix is always a diagonal matrix"""
        Kx, Ky, I = self.Kx, self.Ky, self.get_eye()
        Q = [[Kx@Ky, ur*er*I - Kx@Kx], [Ky@Ky - ur*er*I, -Ky@Kx]]
        return 1/ur * spa.bmat(Q, format='csc', dtype=self.dtype)

    def homo_decompose(self, er=1.+0j, ur=1.+0j):
        """homo W Lam V are all diagonal matrices"""
        W = self.get_eye(2)
        Kz = self.get_Kz(er, ur)
        Lam = spa.bmat([[1j*Kz, None], [None, 1j*Kz]], format='csc', dtype=self.dtype)
        setattr(Lam, 'diag', Lam.diagonal())
        Q = self._homo_Qmatrix(er, ur)
        V = Q @ inv(Lam)
        return W, Lam, V

    @log("Computing general Q matrix on {}".format(torch.cuda.get_device_name('cuda')))
    def _general_Qmatrix(self, er:MAT, ur:MAT) -> spa.spmatrix:
        """Usually general Q matrix is block-wise half sparse matrix, thus we treat Q as a dense matrix"""
        Kx, Ky = self.Kx, self.Ky
        if isinstance(ur, torch.Tensor):
            Kx_cuda = totorch(Kx)
            Ky_cuda = totorch(Ky)
            er_cuda = totorch(er)
            lu_ur = torch.linalg.lu_factor(ur)
            urKx = torch.lu_solve(Kx_cuda, *lu_ur)
            urKy = torch.lu_solve(Ky_cuda, *lu_ur)
            Q = [[Kx_cuda@urKy, er_cuda - Kx_cuda@urKx], [Ky_cuda@urKy - er_cuda, -Ky_cuda@urKx]]
        else:
            urKx = div(ur, Kx)
            urKy = div(ur, Ky)
            if isinstance(er, torch.Tensor):
                Q = [[Kx@urKy, sub(er, Kx@urKx)], [sub(Ky@urKy, er), -Ky@urKx]]
            else:
                Q = [[Kx@urKy, er-Kx@urKx], [Ky@urKy-er, -Ky@urKx]]
        return block(Q)

    @log("Computing general P matrix on {}".format(torch.cuda.get_device_name('cuda')))
    def _general_Pmatrix(self, er:Union[Number, torch.Tensor], ur:Union[Number, torch.Tensor]) -> spa.spmatrix:
        """Usually general P matrix is a dense matrix"""
        Kx, Ky = self.Kx, self.Ky
        if isinstance(er, torch.Tensor):
            Kx_cuda = totorch(Kx)
            Ky_cuda = totorch(Ky)
            ur_cuda = totorch(ur)
            lu_er = torch.linalg.lu_factor(er)
            erKx = torch.lu_solve(Kx_cuda, *lu_er)
            erKy = torch.lu_solve(Ky_cuda, *lu_er)
            P = [[Kx_cuda@erKy, ur_cuda - Kx_cuda@erKx], [Ky_cuda@erKy - ur_cuda, -Ky_cuda@erKx]]
        else:
            erKy = div(er, Kx)
            erKx = div(er, Ky)
            if isinstance(ur, torch.Tensor):
                P = [[Kx@erKy, sub(ur, Kx@erKx)], [sub(Ky@erKy, ur), -Ky@erKx]]
            else:
                P = [[Kx@erKy, ur - Kx@erKx], [Ky@erKy - ur, -Ky@erKx]]
        return block(P)

    @log("Eigendecomposition")
    def general_decompose(self, er:Union[Number, torch.Tensor], ur:Union[Number, torch.Tensor]) -> spa.spmatrix:
        """W and V are dense matrix, Lam is sparse diagonal matrix"""
        omg2, Q = self._prepare_eig(er, ur)
        print(f"Mem usage of Q: {Q.element_size()*Q.nelement()/1024**2}MB, mem_allocated:{torch.cuda.memory_allocated()/1024**2}MB")
        """ This line below is very expensive """
        lam2, W = eig(omg2)  # P,Q here must be dense matrix as it is not possible to calculate full eigenvectors of a sparse matrix?
        print(f"Mem usage of W: {W.element_size()*W.nelement()/1024**2}MB, mem_allocated:{torch.cuda.memory_allocated()/1024**2}MB")
        lam = np.sqrt(lam2)
        Lam = spa.diags(lam, format='csc', dtype=self.dtype)
        setattr(Lam, 'diag', lam)
        Lam_inv = inv(Lam)
        V = matmul(matmul(Q, W), Lam_inv)
        print(f"Mem usage of V: {V.element_size()*V.nelement()/1024**2}MB, mem_allocated:{torch.cuda.memory_allocated()/1024**2}MB")
        return W, Lam, V

    @log("Preparing omg2 matrix")
    def _prepare_eig(self, er, ur):
        P = self._general_Pmatrix(er, ur)
        Q = self._general_Qmatrix(er, ur)
        omg2 = totorch(matmul(P, Q), device='cpu')
        return omg2, Q




@log(f"Computing fft on {torch.cuda.get_device_name('cuda')}")
def fft2(arr:np.ndarray) -> np.ndarray:
    Nxy = np.product(arr.shape)  # total number of points in real space
    arr_torch = numpy2torch(arr)
    Arr_torch = torch.fft.fft2(arr_torch)/Nxy  # Fourier tranform of arr (normalised)
    Arr = torch2numpy(Arr_torch)
    return Arr

@log("Truncating the Fourier Series of the sturcture", clean=False)
def roll(arr:np.ndarray, Mx:int, My:int) -> np.ndarray:
    """arr: the array to be transformed; Mx, My: range of frequency space (-Mx..Mx) (-My..My)"""
    Arr = np.roll(arr, (Mx, My), axis=(0,1))[:2*Mx+1, :2*My+1]  # truncate the wanted frequencies
    return Arr

@log("Constructing the convolution matrix", clean=False)
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

"""Functions involving Smatrix operation"""
@log("Computing reflection side S-Matrix")
def get_refSmatrix(Nmodes, V:spa.spmatrix, V0:spa.spmatrix) -> SMatrix:
    Wterm = spa.identity(2*Nmodes, dtype=V.dtype, format='csc')  # W0_inv @ W, but both matrix are identity
    Vterm = div(V0, V)  # sparse
    A = Wterm + Vterm  # sparse
    B = Wterm - Vterm  # sparse
    A_inv = inv(A)  # sparse
    S11 = -A_inv @ B  # sparse
    S12 = 2 * A_inv  # sparse
    S21 = 0.5 * ( A - B@A_inv@B )  # 0.5*(A-B@A_inv@B)  # sparse
    S22 = B @ A_inv  # sparse
    return SMatrix(S11, S12, S21, S22)

@log("Computing transmission side S-Matrix")
def get_trmSmatrix(Nmodes, V:spa.spmatrix, V0:spa.spmatrix) -> SMatrix:
    Wterm = spa.identity(2*Nmodes, dtype=V.dtype, format='csc')  # W0_inv @ W, but both matrix are identity
    Vterm = div(V0, V)  # sparse
    A = Wterm + Vterm  # sparse
    B = Wterm - Vterm  # sparse
    A_inv = inv(A)  # sparse
    S22 = -A_inv @ B  # sparse
    S21 = 2 * A_inv  # sparse
    S12 = 0.5 * ( A - B@A_inv@B )  # 0.5*(A-B@A_inv@B)  # sparse
    S11 = B @ A_inv  # sparse
    return SMatrix(S11, S12, S21, S22)

@log("Computing homogeneous layer S-Matrix")
def get_homo_Smatrix(Nmodes, Lam:spa.csc_matrix, V:spa.spmatrix, V0:spa.spmatrix, k0, thick=0.) -> SMatrix:
    """if is_homo: all components of S is diagonal sparse matrix, else: all component is dense matrix"""
    Wterm = spa.identity(2*Nmodes, dtype=Lam.dtype, format='csc')  # W0_inv @ W, but both matrix are identity
    Vterm = div(V, V0)  # sparse
    A = Wterm + Vterm  # sparse
    B = Wterm - Vterm  # sparse
    X = expm( -k0*thick * Lam)  # sparse
    BA_inv = rdiv(A, B)  # sparse
    D_inv = spLA.inv(A - X @ BA_inv @ X @ B)  # sparse
    S11 = D_inv @ (X@BA_inv@X@A - B)  # sparse
    S12 = D_inv @ (X@(A - BA_inv@B))  # sparse
    S21 = S12  # sparse
    S22 = S11  # sparse
    return SMatrix(S11, S12, S21, S22)

@log(f"Computing general layer S-Matrix on {torch.cuda.get_device_name('cuda')}")
def get_Smatrix(W:MAT, Lam:spa.csc_matrix, V:MAT, W0:spa.spmatrix, V0:spa.spmatrix, k0, thick=0.) -> SMatrix:
    """if is_homo: all components of S is diagonal sparse matrix, else: all component is dense matrix"""
    Wterm = div(W, W0)
    Vterm = div(V, V0)
    A = add(Wterm, Vterm)  # MAT
    B = sub(Wterm, Vterm)  # MAT
    Wterm, Vterm = None, None
    torch.cuda.empty_cache()
    X = totorch(expm( -k0*thick * Lam))  # sparse
    BA_inv = rdiv(A, B)  # MAT
    D_lu = torch.linalg.lu_factor(A - X@BA_inv@X@B)
    S11 = torch.lu_solve((X@BA_inv@X@A - B), *D_lu)  # MAT
    S12 = torch.lu_solve(X@(A - BA_inv@B), *D_lu)  # MAT
    S21 = S12  # MAT
    S22 = S11  # MAT
    return SMatrix(S11, S12, S21, S22)

@log("Computing global S-Matrix")
def get_total_Smat(*Smats):
    new_Smats = []
    last = None
    print("Compressing S-matrices")
    for n, Smat in enumerate(Smats):
        if last is None:
            last = Smat
            if not Smat.is_sparse:
                new_Smats.append(Smat)
                last = None
        else:
            if Smat.is_sparse:
                last = last * Smat
            else:
                new_Smats.append(last)
                new_Smats.append(Smat)
                last = None
    if last is not None:
        new_Smats.append(last)
    tot = None
    print("Computing total S-matrix")
    for Smat in new_Smats:
        tot = tot * Smat if tot is not None else Smat
    return tot