from .conv_kernel import CConvKernel
import numpy as np

class CConvKernelMovingAverage(CConvKernel):
    def kernel_mask(self):
        self._mask = np.ones(self.kernel_size)
        self._mask /= self.kernel_size