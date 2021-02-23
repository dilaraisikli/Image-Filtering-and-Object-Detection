# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
from math import pi, sqrt, exp
from scipy.ndimage import convolve

"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):

   
    #Set the Parameters:
    x  = np.arange(-3*int(sigma), 3*int(sigma) + 1)
    Gx = np.zeros(x.shape)
    
    # Build the array:

    for i in range(x.size):

        t1 = 1 / ( sigma * sqrt(2 * pi) )
        t2 = exp( -x[i]**2 / (2 * sigma**2) )
        g = t1 * t2
        Gx[i] = g
    
    return Gx, x


"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""
def gaussianfilter(img, sigma):
    
    # Generate the Kernel:
    Gx, x = gauss(sigma)  
    
    # Applay 1D filter on both axis:
        
    f1 = np.apply_along_axis(convolve, 0, img, weights = Gx, mode = 'constant')
    smooth_img = np.apply_along_axis(convolve, 1, f1, weights = Gx, mode = 'constant')
            
    return smooth_img


"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):
    
    #Set the Parameters:
    x  = np.arange(-3*sigma, 3*sigma + 1)
    Dx = np.zeros(x.size)

    # Build the array:
    for i in range(x.size):
        t1 = - (1 / ( (sigma**3) * sqrt(2*pi) ))
        t2 = x[i] * exp( -x[i]**2 / (2 * sigma**2) )
        g = t1 * t2
        Dx[i] = g
        
    return Dx, x


def gaussderiv(img, sigma):

    # Generate the Kernel:
    Dx, x = gaussdx(sigma)  
    
    # Implement convolution:
    imgDx = np.apply_along_axis(convolve, 0, img, weights = Dx, mode = 'constant')
    imgDy = np.apply_along_axis(convolve, 1, img, weights = Dx, mode = 'constant')
    
    return imgDx, imgDy

