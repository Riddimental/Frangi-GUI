import math
import cv2
import nibabel as nib
#np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil
import numpy as np
import scipy.ndimage as ndi
from skimage import filters
from scipy.linalg import eigvals, norm
from itertools import combinations_with_replacement



def delete_temp():
   # Delete the contents of the temp folder
   folder_path = "temp"

   # Check if the folder exists
   if os.path.exists(folder_path):
      # Iterate over the files in the folder and delete them
      for filename in os.listdir(folder_path):
         file_path = os.path.join(folder_path, filename)
         try:
               if os.path.isfile(file_path):
                  os.unlink(file_path)
               elif os.path.isdir(file_path):
                  shutil.rmtree(file_path)
         except Exception as e:
               print(f"Failed to delete {file_path}. Reason: {e}")
   else:
      print(f"The folder {folder_path} does not exist.")

def divide_nonzero(array1, array2):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)

def thresholding(data, threshold):
   transformed_image = data > threshold
   #print(transformed_image)
   return transformed_image
   
def thresholding2d(data, threshold):
   transformed_image = data > threshold
   plt.imsave("temp/plot.jpeg", transformed_image, cmap = 'gray')

def gaussian_preview(data, intensity):
   data = ndi.gaussian_filter(data, intensity)
   plt.imsave("temp/plot.jpeg", data, cmap='gray')
   #plt.close()

def gaussian3d(data3D, intensity):
   kernel_size = max(1, math.trunc(intensity))
   if kernel_size % 2 == 0:
      kernel_size += 1  # Make it odd if it's even
   data = ndi.gaussian_filter(data3D, intensity/3)
   return data

def sci_frangi(image, scale_range=(1, 10), alpha=1, beta=0.5, steps=2, cval=1):
   processed = filters.frangi(image, sigmas=np.arange(scale_range[0], scale_range[1], steps), alpha=alpha, beta=beta, cval=cval)
   return processed

def my_frangi_filter(image, scale_range=(1, 10), alpha=1, beta=0.5, steps=2, cval=1, black_vessels=True):
   # Ensure input image is of type float64
   image = image.astype(np.float64)
   
   if not black_vessels:
        image = -image

   division = (scale_range[1] - scale_range[0]) / steps
   vesselness = np.zeros_like(image)
   print('Scales from', scale_range[0], 'to', scale_range[1], 'in', steps, 'steps.')

   scale = scale_range[0]
   while scale <= scale_range[1]:
      print('Current scale is:', scale)
      eigenvalues, cvar = compute_hessian_return_eigvals(image, sigma=scale)
      vesselness += compute_vesselness(eigenvalues, alpha, beta, cvar)
      scale += division
   
   
   print("Frangi filter applied.")
   return vesselness

def compute_hessian_return_eigvals(image, sigma=1):
   """Compute the Hessian via convolutions with Gaussian derivatives."""
   float_dtype = np.float64
   image = image.astype(float_dtype, copy=False)

   if np.isscalar(sigma):
      sigma = (sigma,) * 3  # For 3D images

   # Gaussian smoothing using ndi.gaussian_filter
   smoothed_image = ndi.gaussian_filter(image, sigma=sigma, mode='reflect')

   # Gradients
   gradients = np.gradient(smoothed_image)

   # Structure tensors (Hessian)
   Hessian_matrix = []
   for i in range(3):  # 3D image, so we iterate over each dimension
      D1 = np.gradient(gradients[i], axis=i)
      D2 = np.gradient(D1, axis=i)
      Hessian_matrix.append(D2)
      
   print('hessian ',np.shape(Hessian_matrix))
   # Compute the norm of the structure tensors for cval
   cval = norm(Hessian_matrix) / 2
   print('cval ',cval)
   # Compute eigenvalues
   eig_vals = compute_eig_vals(Hessian_matrix)

   return eig_vals, cval

def compute_eig_vals(H_elems):
   """Compute eigenvalues from the upper-diagonal entries of a symmetric matrix."""
   M00, M01, M11 = H_elems
   eigs = np.empty((3, *M00.shape), M00.dtype)
   eigs[0] = (M00 + M11) / 2
   hsqrtdet = np.sqrt(M01 ** 2 + ((M00 - M11) / 2) ** 2)
   eigs[1] = (M00 + M11) / 2 + hsqrtdet
   eigs[2] = (M00 + M11) / 2 - hsqrtdet
   print("eigs ",eigs.shape)
   return eigs

def compute_vesselness(eigvals, alpha, beta, c):
   """Compute vesselness measure from Hessian elements."""
   # Extract Hessian elements
   eigvals = np.take_along_axis(eigvals, abs(eigvals).argsort(0), 0)
   lambda1= eigvals[0]
   lambda2, lambda3 = np.maximum(eigvals[1:], 1e-10)

   # Compute vesselness measure
   Ra = divide_nonzero(np.abs(lambda2), np.abs(lambda3))
   Rb = divide_nonzero(np.abs(lambda1), np.sqrt(np.abs(np.multiply(lambda2, lambda3))))
   S2 = np.sqrt(np.square(lambda1) + np.square(lambda2) + np.square(lambda3))

   vesselness = (1 - np.exp(-Ra**2 / (2 * alpha**2))) * np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S2**2 / (2 * c**2)))
   return vesselness
