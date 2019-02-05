import numpy
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy import misc, ndimage
import math

def gaussianFilter(image, sigma, high=True):
    rows, columns = image.shape
    # print(rows, columns)
    centerX = int(rows / 2)
    centerY = int(columns / 2)

    arr = []

    for i in range(rows):
        row = []
        for j in range(columns):
            val = math.exp(-1 * ((i - centerX)**2 + (j - centerY)**2) / (2 * (sigma**2)))
            row.append(1 - val) if high else row.append(val)

        arr.append(row)
    # print(numpy.array(arr).shape)

    return numpy.array(arr)

def DFT(image, filter):
    dft = fftshift(fft2(image))

    filtered = dft * filter
    return ifft2(ifftshift(filtered))

def highPassFilter(highFreqImg, alpha):
    return DFT(highFreqImg, gaussianFilter(highFreqImg, alpha, high=True))

def lowPassFilter(lowFreqImg, beta):
    return DFT(lowFreqImg, gaussianFilter(lowFreqImg, beta, high=False))

def hybridImage(highFreqImg, lowFreqImg, alpha, beta):
    highPassImg = highPassFilter(highFreqImg, alpha)
    lowPassImg = lowPassFilter(lowFreqImg, beta)

    return highPassImg + lowPassImg

if __name__ == "__main__":
    einstein = ndimage.imread("HW1_Q1/einstein.bmp", flatten=True)
    marilyn = ndimage.imread("HW1_Q1/marilyn.bmp", flatten=True)

    hybrid_img = hybridImage(einstein, marilyn, 25, 10)
    misc.imsave("einstein-marilyn.png", numpy.real(hybrid_img))
