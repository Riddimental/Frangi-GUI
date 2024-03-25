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
from scipy.linalg import eigvals

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

def sci_frangi(image3D):
   processed = filters.frangi(image3D)
   return processed

def hessian_eigen(image, sigma):
   # Gaussian smoothing
   smoothed_image = ndi.gaussian_filter(image, sigma)

   # Gradients
   gradients = np.gradient(smoothed_image)

   # Second derivatives
   Dxx = np.gradient(gradients[0], axis=0)
   Dyy = np.gradient(gradients[1], axis=1)
   Dzz = np.gradient(gradients[2], axis=2)

   Dxy = np.gradient(gradients[0], axis=1)
   Dxz = np.gradient(gradients[0], axis=2)
   Dyz = np.gradient(gradients[1], axis=2)

   # Construct the Hessian matrix for each voxel
   H = np.zeros((image.shape[0], image.shape[1], image.shape[2], 3, 3))
   H[..., 0, 0] = Dxx
   H[..., 1, 1] = Dyy
   H[..., 2, 2] = Dzz
   H[..., 0, 1] = H[..., 1, 0] = Dxy
   H[..., 0, 2] = H[..., 2, 0] = Dxz
   H[..., 1, 2] = H[..., 2, 1] = Dyz

   # Compute eigenvalues of the Hessian matrix
   eigenvalues = np.linalg.eigvalsh(H)
   

   return eigenvalues

def my_frangi_filter(image, scale_range=(1, 10), alpha=1, beta=0.5, steps=2, c=1):
   # Ensure input image is of type float64
   #image = image.astype(np.float64)
   
   division = (scale_range[1] - scale_range[0])/steps

   # Initialize output vesselness image as floating-point
   vesselness = np.zeros_like(image)
   
   print('scales from ', scale_range[0] , ' to ', scale_range[1], ' in ', steps, ' steps.')

   scale = scale_range[0]
   while scale <= scale_range[1]:
      # Compute Hessian matrix components
      # Your code to compute Hessian matrix components goes here
      print('current scale is: ', scale)
      
      eigenvalues = hessian_eigen(image, scale)
      lambda1 = eigenvalues[..., 0]
      lambda2 = eigenvalues[..., 1]
      lambda3 = eigenvalues[..., 2]

      epsilon = 1e-6
      # Compute vesselness measure for the current scale in 3D
      Ra = np.abs(lambda2) / np.abs(lambda3 + epsilon)
      Rb = np.abs(lambda1) / (np.sqrt(np.abs(lambda2 * lambda3)) + epsilon)
      S2 = np.sqrt(lambda2**2 + lambda3**2)
      
      vesselness = (1 - np.exp(-1*(Ra**2 / (2 * alpha**2)))) * np.exp(-1*(Rb**2 / (2 * beta**2))) * (1 - np.exp(-1*(S2**2 / (2 * c**2))))

      # Increment scale by the step size
      scale += division
      
   print("frangi aplied")
   return vesselness
