from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np

class CConvKernel(ABC):
    def __init__(self, kernel_size = 3):
        self._mask = None
        self.kernel_size = kernel_size
        super().__init__()

    @property
    def mask(self):
        return deepcopy(self._mask) #doing this wo call the mask from outsid cannot change the single elements

    @property
    def kernel_size(self):
        return self._kernel_size

    @kernel_size.setter
    def kernel_size(self, val):
        if not isinstance(val, int):
            raise TypeError("Kernel size must be a integer number")
        if (val % 2 == 0) or (val <= 0):
            raise ValueError("Kernel size must be odd and positive")
        self._kernel_size = val
        self.kernel_mask()

    @abstractmethod
    def kernel_mask(self):
        pass

    def kernel(self, x):
        """calculate the convolution

        Args:
            x (int): image vector

        Returns:
            x_conv (int): convoluted image
        """
        #x.shape = (something, )
        x_conv = deepcopy(x)
        h = int((self.kernel_size - 1)/2)
        for i in range( h , x.shape[0] - h):
            # Border elements remain unchanged
            x_conv[i] = np.dot(x[i - h: i + h + 1], self._mask)
        return x_conv
        
