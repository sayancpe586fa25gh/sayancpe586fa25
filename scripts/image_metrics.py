import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def laplacian_variance(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def tenengrad(img):
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    return np.mean(gx**2 + gy**2)

def high_freq_energy(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return np.mean(np.abs(fshift))

def local_std(img):
    return np.std(img)

def glcm_contrast(img):
    glcm = graycomatrix(img, [1], [0], 256, symmetric=True, normed=True)
    return graycoprops(glcm, 'contrast')[0,0]
