Overview

This project implements several core image-processing filters:

1D Gaussian kernel

2D Gaussian smoothing using separability

Gaussian derivative (DoG) filters

Impulse response visualization

Edge detection using derivatives of Gaussian

All code is implemented manually using NumPy, SciPy, and Pillow — no OpenCV or high-level libraries.

What This Project Covers
1. 1D Gaussian Kernel

Implemented using the analytical formula:

G(x) = 1 / (sigma * sqrt(2π)) * exp( -x² / (2 * sigma²) )


Defined over the range:

x ∈ [ -3*sigma , +3*sigma ]


A plot is generated using Matplotlib.

2. 2D Gaussian Smoothing (Using Separable Convolution)

A 2D Gaussian filter is computed by two 1D filters:

G(x, y) = G(x) * G(y)


Filtering is done by:

Convolve the image with the 1D Gaussian along rows

Convolve the result with the same kernel along columns

This is more efficient and mathematically equivalent to full 2D convolution.

3. Derivative-of-Gaussian (DoG) Filter

The 1D derivative of Gaussian is implemented via:

D(x) = -(x / (sigma³ * sqrt(2π))) * exp( -x² / (2 * sigma²) )


The script also plots Gaussian vs. Gaussian derivative.

4. Impulse Response Experiments

A 27×27 image with a single white pixel is used to visualize:

Gaussian blur

First derivative in x

First derivative in y

Combined filters such as:

G * Gᵀ
G * Dᵀ
D * Gᵀ
D * Dᵀ

5. Edge Detection Using DoG

Given an image graf.png, we compute:

Ix = I convolved with Dx     # horizontal edges
Iy = I convolved with Dy     # vertical edges


Gradient magnitude:

|∇I| = sqrt( Ix² + Iy² )


This highlights edge strength independent of direction.

Implemented Functions
gauss(sigma)

Returns a 1D Gaussian kernel and its x-values.

gaussianfilter(img, sigma)

Applies smoothing using two 1D Gaussian convolutions.

gaussdx(sigma)

Computes the 1D derivative of Gaussian.

gaussderiv(img, sigma)

Returns:

imgDx — derivative along x

imgDy — derivative along y

Outputs Produced

Plot of 1D Gaussian

Original vs. smoothed image

Gaussian vs. derivative plot

Impulse response visualizations for multiple filter combinations

Edge images:

Horizontal (Dx)

Vertical (Dy)

Gradient magnitude

How to Run

Install dependencies:

pip install numpy scipy pillow matplotlib


Run the script:

python filter.py


graf.png must be in the same directory.
