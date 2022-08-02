from tkinter import W
import numpy as np
import scipy.sparse as spa
from numpy import linalg as LA
from scipy.sparse import linalg as spLA
import torch
from time import process_time
from functools import wraps
import tracemalloc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def mem_profile():
    gpu = torch.cuda.memory_allocated()
    gpu_peak = torch.cuda.max_memory_allocated()
    cpu, cpu_peak = tracemalloc.get_traced_memory()
    return cpu, cpu_peak, gpu, gpu_peak

def mem_diff(old, new):
    cpu = (new[0] - old[0])
    cpu_peak = (new[1] - old[1])
    gpu = (new[2] - old[2])
    gpu_peak = (new[3] - old[3])
    return cpu, cpu_peak, gpu, gpu_peak

def reset_profile():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tracemalloc.reset_peak()

def auto_unit(val, unit=1000, prefix=' KMGTPEZYyzafpn\u03bcm', dim=1, order=0, top=8, low=-8):
    level = unit ** (dim*order)
    if val == 0:
        return "0 "
    if order < low or order > top:
        return "{:.3E} ".format(val)
    if level/10 <= abs(val) < level*unit:
        return "{:.3f}{}".format(val/level, prefix[order])
    elif np.abs(val) > level:
        return auto_unit(val, unit=unit, prefix=prefix, dim=dim, order=order+1)
    elif np.abs(val) < level:
        return auto_unit(val, unit=unit, prefix=prefix, dim=dim, order=order-1)

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        reset_profile()
        mem_0 = mem_profile()
        start = process_time()
        time = 0.
        num = 0
        while time <= 0.2:
            try:
                res = func(*args, **kwargs)
            except (TypeError, RuntimeError) as error:
                print(f"Running {func.__name__} failed: {error}")
            end = process_time()
            time = (end - start)  # in s
            num += 1
        time /= num
        mem_1 = mem_profile()
        mem = np.max(mem_diff(mem_0, mem_1))
        tracemalloc.stop()
        msg = "Running {} success with time {}s and max_memory {}B"
        time_msg = auto_unit(time)
        mem_msg = auto_unit(mem, unit=1024, top=3, low=0)
        print(msg.format(func.__name__, time_msg, mem_msg))
        return time, mem, res
    return wrapper

"""Available Setup functions, naming scheme: setup_name1_name2_..."""
def setup_numpy_torch(size):
    A = np.random.rand(size, size)
    B = torch.tensor(A, device='cuda')
    return A, B

"""Available choice for functions"""
@profile
def pinv(*args):
    A = args[0]
    if isinstance(A, np.ndarray):
        return LA.pinv(A)
    if isinstance(A, torch.Tensor):
        return torch.linalg.pinv(A)
    if isinstance(A, spa.spmatrix):
        raise TypeError("pinv of a sparse matrix is not supported by SciPy")
    raise TypeError("Invalid input matrix type!")

@profile
def inv(*args):
    A = args[0]
    if isinstance(A, np.ndarray):
        return LA.inv(A)
    if isinstance(A, torch.Tensor):
        return torch.linalg.inv(A)
    if isinstance(A, spa.spmatrix):
        return spLA.inv(A)
    raise TypeError("Invalid input matrix type!")

@profile
def eig(*args):
    A = args[0]
    if isinstance(A, np.ndarray):
        return LA.eig(A)
    if isinstance(A, torch.Tensor):
        return torch.linalg.eig(A)
    if isinstance(A, spa.spmatrix):
        raise TypeError("Full eigendecomposition of a sparse matrix is not supported by SciPy")
    raise TypeError("Invalid input matrix type!")

@profile
def mul(*args):
    A, B = args[0:2]
    if type(A) != type(B):
        if isinstance(A, (np.ndarray, spa.spmatrix)) and isinstance(B, (np.ndarray, spa.spmatrix)):
            return np.array(A @ B)
        raise TypeError("Matrix A and B should have same type!")
    if isinstance(A, (np.ndarray, torch.Tensor, spa.spmatrix)):
        return A @ B
    raise TypeError("Invalid input matrix type!")

def _div(*args):
    """ Calculate A_inv @ B """
    A, B = args[0:2]
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return torch.linalg.solve(A, B)
    if isinstance(A, np.ndarray):
        if isinstance(B, np.ndarray):
            return LA.solve(A, B)
        if isinstance(B, spa.spmatrix):
            return LA.inv(A) @ B
    if isinstance(A, spa.spmatrix) and isinstance(B, (np.ndarray, spa.spmatrix)):
        return spLA.spsolve(A, B)
    raise TypeError("Invalid input matrix type!")

@profile
def div(*args):
    return _div(*args)

@profile
def rdiv(*args):
    """ Calculate A @ B_inv, this could be done by  solve(B.T, A.T).T """
    A, B = args[0:2]
    a = B.T
    b = A.T
    res = _div(a, b)
    return res.T
    

class Benchmark:
    def __init__(self, piplines, setup, maxsize=1000, minsize=10, total=50):
        self.input = np.geomspace(minsize, maxsize, total, endpoint=False, dtype=int)
        self.setup = setup
        matrix_names = setup.__name__.split('_')[1:]
        self.ntype = len(matrix_names)
        self.piplines = piplines
        self.time = np.zeros((self.ntype, len(self.piplines), total))
        self.mem = np.zeros((self.ntype, len(self.piplines), total))
        self.labels = [
            [ '_'.join([pre, func.__name__]) for func in self.piplines ]
            for pre in matrix_names
        ]

    def run(self):
        for n, size in enumerate(self.input):
            print(f"Running for size {size}")
            As = self.setup(size)
            Bs = self.setup(size)
            for i, A in enumerate(As):
                B = Bs[i]
                for j, func in enumerate(self.piplines):
                    time, mem, res = func(A, B)
                    self.time[i, j, n] = time
                    self.mem[i, j, n] = mem
                    del res
                    torch.cuda.empty_cache()
            del As, Bs
            torch.cuda.empty_cache()
    
    def save(self, filename):
        np.savez(
            filename,
            input = self.input,
            labels = self.labels,
            time = self.time,
            mem = self.mem,
        )


class BenchmarkData(Benchmark):
    def __init__(self, filename):
        data = np.load(filename)
        self.input = data['input']
        self.labels = data['labels']
        self.time = data['time']
        self.mem = data['mem']
    
    def run(self):
        pass


def plot_benchmark(bm:Benchmark):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ylabels = ['Time usage (ms)', 'Memory usage (MB)']
    data = [bm.time*1e3, bm.mem/1024**2]
    marks = 'ox+D.s*'
    colors = list(mcolors.TABLEAU_COLORS.values())
    for n, ax in enumerate(axes):
        ax.set_xlabel('Matrix size')
        ax.set_ylabel(ylabels[n]) 
        for i in range(bm.time.shape[0]):
            mark = marks[i]
            for j in range(bm.time.shape[1]):
                color = colors[j]
                ax.loglog(bm.input, data[n][i, j, :], 
                    marker=mark, c=color, lw=1.5, 
                    label=bm.labels[i][j] if n == 0 else None
                    )
    fig.legend(loc=7, bbox_to_anchor=(1,0.5))
    return fig

    

if __name__ == '__main__':
    piplines = [ pinv, inv, eig, mul, div, rdiv ]
    setup = setup_numpy_torch
    bm = Benchmark(piplines, setup, maxsize=100, minsize=10, total=5)
    bm.run()
    bm.save('benchmark')



