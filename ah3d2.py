import ctypes
import numpy as np

lib = ctypes.cdll.LoadLibrary('./libAH3D2.so')

class AH(object):
    def __init__(self, M, N, K, H, hx, hy,hz):
        lib.AH_new.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
        np.ctypeslib.ndpointer(dtype=np.float64,ndim=4,shape = (M*N*K,6,3,3)), ctypes.c_double,ctypes.c_double,ctypes.c_double]
        lib.AH_new.restype = ctypes.c_void_p

        lib.AH_Row.argtypes = [ctypes.c_void_p]
        lib.AH_Row.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*K*19,))

        lib.AH_Col.argtypes = [ctypes.c_void_p]
        lib.AH_Col.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_int, shape = (M*N*K*19,))

        lib.AH_Val.argtypes = [ctypes.c_void_p]
        lib.AH_Val.restype = np.ctypeslib.ndpointer(dtype=ctypes.c_double, shape = (M*N*K*19,))
        
        lib.AH_delete.argtypes = [ctypes.c_void_p]

        self.obj = lib.AH_new(M, N, K, H, hx, hy, hz)


    def Row(self):
        return (lib.AH_Row(self.obj))

    def Col(self):
        return (lib.AH_Col(self.obj))
    
    def Val(self):
        return (lib.AH_Val(self.obj))

    def __del__(self):
        return lib.AH_delete(self.obj)