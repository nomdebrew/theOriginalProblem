## ---------------------------- ##
## 
## sample_student.py
##
## Example student submission for programming challenge. A few things: 
## 1. Before submitting, change the name of this file to your firstname_lastname.py.
## 2. Be sure not to change the name of the method below, classify.py
## 3. In this challenge, you are only permitted to import numpy and methods from 
##    the util module in this repository. Note that if you make any changes to your local 
##    util module, these won't be reflected in the util module that is imported by the 
##    auto grading algorithm. 
## 4. Anti-plagarism checks will be run on your submission
##
##
## ---------------------------- ##


import numpy as np
#It's kk to import whatever you want from the local util module if you would like:
#from util.X import ... 

def convert_to_grayscale(im):
    '''
    Convert color image to grayscale.
    Args: im = (nxmx3) floating point color image scaled between 0 and 1
    Returns: (nxm) floating point grayscale image scaled between 0 and 1
    '''
    return np.mean(im, axis = 2)

def filter_2d(im, kernel):
    '''
    Filter an image by taking the dot product of each 
    image neighborhood with the kernel matrix.
    Args:
    im = (H x W) grayscale floating point image
    kernel = (M x N) matrix, smaller than im
    Returns: 
    (H-M+1 x W-N+1) filtered image.
    '''

    M = kernel.shape[0] 
    N = kernel.shape[1]
    H = im.shape[0]
    W = im.shape[1]
    
    filtered_image = np.zeros((H-M+1, W-N+1), dtype = 'float64')
    
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            image_patch = im[i:i+M, j:j+N]
            filtered_image[i, j] = np.sum(np.multiply(image_patch, kernel))
            
    return filtered_image

def classify(im):
    #blur kernel
    blur_size = 3
    blur_kernel = (1/blur_size**2)*np.ones((blur_size,blur_size))
    blur_kernel
    #Implement Sobel kernels as numpy arrays
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    grayed = convert_to_grayscale(im)
    blured = filter_2d(grayed, blur_kernel)
    Gx = filter_2d(blured, Kx)
    Gy = filter_2d(blured, Ky)
    
    #Compute Gradient Magnitude and Direction:
    G_magnitude = np.sqrt(Gx**2+Gy**2)
    #G_direction = np.arctan2(Gy, Gx)
    edges = G_magnitude > 200
    y_coords, x_coords = np.where(edges)
    y_coords_flipped = edges.shape[1] - y_coords
    
    #set up accumulator
    phi_bins = 64
    theta_bins = 64
    accumulator = np.zeros((phi_bins, theta_bins))
    rho_min = -edges.shape[0]
    rho_max = edges.shape[1]
    theta_min = 0
    theta_max = np.pi

    #Compute the rho and theta values for the grids in our accumulator:
    rhos = np.linspace(rho_min, rho_max, accumulator.shape[0])
    thetas = np.linspace(theta_min, theta_max, accumulator.shape[1])
    
    for i in range(len(x_coords)):
        #Grab a single point
        x = x_coords[i]
        y = y_coords_flipped[i]

        #Actually do transform!
        curve_rhos = x*np.cos(thetas)+y*np.sin(thetas)

        for j in range(len(thetas)):
            #Make sure that the part of the curve falls within our accumulator
            if np.min(abs(curve_rhos[j]-rhos)) <= 1.0:
                #Find the cell our curve goes through:
                rho_index = np.argmin(abs(curve_rhos[j]-rhos))
                accumulator[rho_index, j] += 1
    edges = accumulator > 80
    total_edges = sum(edges==1)
    max_accum = np.max(accumulator)
    
    if max_accum>= 180:
        print("max:", np.max(accumulator), 'brick ', end = "")
        return 'brick'
    elif max_accum >=80:
        print("max:", np.max(accumulator), 'cylinder ', end = "")
        return 'cylinder'
    else:
        print("max:", np.max(accumulator), 'ball ', end = "")
        return 'ball'