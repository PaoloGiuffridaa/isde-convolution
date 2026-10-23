import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
src = root / 'src'
sys.path.insert(0, str(src))

from conv_1d_kernels import (
    CConvKernelMovingAverage,
    CConvKernelTriangle,
    CConvKernelCombo,
)
import numpy as np
import pytest

def test_moving_average():
    kernel_mov_avg = CConvKernelMovingAverage()
    assert kernel_mov_avg.kernel_size == 3
    assert np.isclose(1, np.sum(kernel_mov_avg.mask))
    assert np.allclose(1/3, kernel_mov_avg.mask)

    with pytest.raises(ValueError):
        kernel_mov_avg.kernel_size = 4
    assert kernel_mov_avg.kernel_size == 3

    kernel_mov_avg.kernel_size = 5
    assert kernel_mov_avg.mask.size == 5
    assert np.allclose(1/5, kernel_mov_avg.mask)

    x_dummy = np.zeros(15)
    assert np.allclose(kernel_mov_avg.kernel(x_dummy), x_dummy)

    x_dummy = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    x_expected = np.array([0., 0., 0., 1/5, 1/5, 1/5, 1/5, 1/5, 0., 0., 0.])
    assert np.allclose(kernel_mov_avg.kernel(x_dummy), x_expected)

def test_triangle():
    kernel_triangle = CConvKernelTriangle()
    assert kernel_triangle.kernel_size == 3
    assert np.isclose(1, np.sum(kernel_triangle.mask))
    x_expected = np.array([0.25,0.5,0.25])
    assert np.allclose( kernel_triangle.mask, x_expected)
    
    with pytest.raises(ValueError):
        kernel_triangle.kernel_size = 4
    assert kernel_triangle.kernel_size == 3
    
    kernel_triangle.kernel_size = 5
    assert kernel_triangle.mask.size == 5
    x_expected = np.array([1/9,2/9,3/9,2/9,1/9])
    assert np.allclose(kernel_triangle.mask, x_expected)
    
    x_dummy = np.zeros(15)
    assert np.allclose(kernel_triangle.kernel(x_dummy), x_dummy)
    
    x_dummy = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
    x_expected = np.array([0., 0., 0., 1/9,2/9,3/9,2/9,1/9, 0., 0., 0.])
    assert np.allclose(kernel_triangle.kernel(x_dummy), x_expected)

def test_combo():
    kernel_mov_avg = CConvKernelMovingAverage()
    kernel_triangle = CConvKernelTriangle()
    kernel_combo = CConvKernelCombo([kernel_mov_avg, kernel_triangle, kernel_mov_avg])

    x_dummy = np.zeros(15)
    x_dummy[7] = 1

    x_expected = np.array([1.,4.,8.,10.,8.,4.,1.]) 
    x_expected /= np.sum(x_expected)

    assert np.allclose(kernel_combo.kernel(x_dummy)[4:11], x_expected)

    print(kernel_combo.kernel(x_dummy))
