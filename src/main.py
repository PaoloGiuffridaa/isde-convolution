import matplotlib.pyplot as plt
import numpy as np
import conv_1d_kernels
from copy import deepcopy

from sklearn.datasets import fetch_openml

def plot_10_img(images, figure_title = None):
    fig, axs = plt.subplots(2, 5, figsize=(10, 5))
    axs = axs.ravel()
    for i in range(10):
        axs[i].imshow(images[i].reshape(28, 28), cmap='gray')
        axs[i].set_title(f'Label: {i}')
        axs[i].axis('off')
    if figure_title is not None:
        plt.suptitle(figure_title)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':\

    print("Starting MNIST download...")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False
    )

    images = mnist.data
    labels = mnist.target

    print("MNIST loaded!")

    print(f'x.shape:{images.shape}')
    print(f'y.shape:{labels.shape}')

    #estraggo una immagine per classe
    img_selected = np.zeros(shape=(10,784))
    unique_labels = np.unique(labels)
    for i in unique_labels:
        img_i = images[np.where(labels == str(i))]
        idx_selected = np.random.randint(img_i.shape[0])
        img_selected[int(i)] =img_i[idx_selected]
   
    plot_10_img(img_selected, figure_title= 'immagini selezionate')

    #applico la convoluzione moving_average
    kernel_mov_avg = conv_1d_kernels.CConvKernelMovingAverage()
    img_conv_mov_avg = deepcopy(img_selected)
    for i in range(img_conv_mov_avg.shape[0]):
        img_conv_mov_avg[i] = kernel_mov_avg.kernel(img_selected[i])
    plot_10_img(img_conv_mov_avg, figure_title= 'immagini convolute con moving average')

    #applico la convoluzione triangle
    kernel_triangle = conv_1d_kernels.CConvKernelTriangle()
    img_conv_triangle = deepcopy(img_selected)
    for i in range(img_conv_triangle.shape[0]):
        img_conv_triangle[i] = kernel_triangle.kernel(img_selected[i])
    plot_10_img(img_conv_triangle, figure_title= 'immagini convolute con triangle')

    #applico la convoluzione combo
    kernel_combo = conv_1d_kernels.CConvKernelCombo([kernel_mov_avg, kernel_triangle, kernel_mov_avg])
    img_conv_combo = deepcopy(img_selected)
    for i in range(img_conv_combo.shape[0]):
        img_conv_combo[i] = kernel_combo.kernel(img_selected[i])
    plot_10_img(img_conv_combo, figure_title= 'immagini convolute con combo')

