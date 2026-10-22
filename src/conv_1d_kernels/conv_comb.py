from copy import deepcopy

class CConvKernelCombo():
    def __init__(self, kernel_list):
        self.kernels = kernel_list

    @property
    def kernels(self):
        return self._kernels
    @kernels.setter
    def kernels(self, kernel_list):
        self._kernels = kernel_list
    
    def kernel(self, x):
        x_conv = deepcopy(x)
        for kern in self._kernels:
            x_conv = kern.kernel(x_conv)
        return x_conv


            