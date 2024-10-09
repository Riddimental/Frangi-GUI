import math
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import scipy.ndimage as ndi
from skimage import filters
from skimage.transform import resize

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

def resample_image(image_data, target_voxel_size):
   current_voxel_sizes = np.array(image_data.header.get_zooms())
   shape = image_data.get_fdata().shape * (current_voxel_sizes/target_voxel_size)
   new_image_data = resize(image_data.get_fdata(), output_shape=shape, mode='constant')
   print("image reshaped from ", image_data.get_fdata().shape , " to ", new_image_data.shape)
   return new_image_data


def mm2voxel(mm, voxel_size):
   num_voxels = mm / voxel_size
   #print("for a voxel size of ",voxel_size," cubic mm, ",mm, "mm's of diameter is ",num_voxels," voxels")
   return num_voxels #returns the diameter

def voxel2mm(num_voxels, voxel_size):
    mm = num_voxels * voxel_size
    #print("for a voxel size of ",voxel_size," cubic mm, ",num_voxels, " voxels is equivalent to ",mm," mm")
    return mm  # returns the diameter in mm

def calculate_noise(input):
   img_data = input
   corner_size = [int(dim * 0.05) for dim in img_data.shape]
   corners = [
      img_data[:corner_size[0], :corner_size[1], :corner_size[2]], # (0,0,0)
      img_data[-corner_size[0]:, :corner_size[1], :corner_size[2]], # (end,0,0)
      img_data[:corner_size[0], -corner_size[1]:, :corner_size[2]], # (0,end,0)
      img_data[:corner_size[0], :corner_size[1], -corner_size[2]:], # (0,0,end)
      img_data[-corner_size[0]:, -corner_size[1]:, :corner_size[2]], # (end,end,0)
      img_data[-corner_size[0]:, :corner_size[1], -corner_size[2]:], # (end,0,end)
      img_data[:corner_size[0], -corner_size[1]:, -corner_size[2]:], # (0,end,end)
      img_data[-corner_size[0]:, -corner_size[1]:, -corner_size[2]:], # (end,end,end)
   ]

   # Combine all the corner samples into one large sample
   large_sample = np.concatenate([corner.flatten() for corner in corners])

   # Calculate the standard deviation of the large sample
   std_deviation_corners = np.std(large_sample)
   
   #print(f'Estimated sigma (σ) for the homogeneous region: {std_deviation_corners}')
   return std_deviation_corners

def intensity_rescale(image, new_min=0, new_max=1):
   """
   Rescale the intensity values of a 3D image array to a specified range.

   Parameters:
   - image: 3D numpy array representing the input image.
   - new_min: Minimum value of the new intensity range (default: 0).
   - new_max: Maximum value of the new intensity range (default: 1).

   Returns:
   - rescaled_image: 3D numpy array containing the rescaled image.
   """
   # Compute the current minimum and maximum intensity values
   current_min = np.min(image)
   current_max = np.max(image)

   # Rescale the intensity values to the new range
   rescaled_image = (image - current_min) / (current_max - current_min)

   # Scale to the new range
   rescaled_image = rescaled_image * (new_max - new_min) + new_min
   
   print("intensity rescaled")

   return rescaled_image

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

def gaussian_preview(data, intensity, root=None):
   data = ndi.gaussian_filter(data, intensity, mode='constant')
   if(root):
      root.after(15, plt.imsave("temp/plot.jpeg", data, cmap='gray'))
   else:
      plt.imsave("temp/plot.jpeg", data, cmap='gray')
   

def gaussian3d(data3D, intensity):
   kernel_size = max(1, math.trunc(intensity))
   if kernel_size % 2 == 0:
      kernel_size += 1  # Make it odd if it's even
   data = ndi.gaussian_filter(data3D, intensity/3, mode='constant')
   return data

def sci_frangi(image, scale_range=(1, 10), alpha=1, beta=0.5, steps=2, cval=1):
   processed = filters.frangi(image, sigmas=np.arange(scale_range[0], scale_range[1], steps), alpha=alpha, beta=beta)
   print("sci-kit frangi applied")
   return processed

def my_frangi_filter(image, sigmas=[1], alpha=1, beta=0.5, black_vessels=True):
   # Ensure input image is of type float64
   image = image.astype(np.float64)
   image /= np.max(image) # image normalized
   
   if black_vessels:
        image = -image

   vesselness = np.zeros_like(image)
   
   for sigma in sigmas:
      print('Current scale:', sigma)
      eigenvalues = compute_hessian_return_eigvals(image, sigma=sigma)
      #print("shapes eigs are ", eigenvalues.shape)
      output = compute_vesselness(eigenvalues, alpha, beta).astype(np.float64)
      vesselness += output
   
   print("Frangi filter applied.")
   return vesselness/np.max(vesselness)

def compute_eig_vals(hessian):
   # Compute eigenvalues
   eigvals = np.linalg.eigvals(hessian.transpose(2, 3, 4, 0, 1))  # Move the 3x3 matrices to the last dimensions

   # Sort eigenvalues in descending order
   sorted_eigvals = -np.sort(-eigvals, axis=-1)

   return sorted_eigvals

def compute_cvals(hessian):
   # Compute the norm of the structure tensors for cval
   norm = np.linalg.norm(hessian, axis=(0, 1)) / 2
   return norm

def compute_hessian_return_eigvals(_3d_image_array, sigma=1):
   """Compute the Hessian via convolutions with Gaussian derivatives."""
   float_dtype = np.float64
   _3d_image_array = _3d_image_array.astype(float_dtype, copy=False)

   if np.isscalar(sigma):
      sigma = (sigma,) * 3  # For 3D images

   # Gaussian smoothing using ndi.gaussian_filter
   smoothed_image = ndi.gaussian_filter(_3d_image_array, sigma=sigma, mode='nearest')
   #print("smoothed image shape is ",smoothed_image.shape)
   
   # Gradients
   gradients = np.gradient(smoothed_image)
   gradients = np.array(gradients)

   #print("gradients shape ", np.shape(gradients))  # gradients shape  (3, 176, 192, 192)

   hessian = np.zeros((3,3,_3d_image_array.shape[0],_3d_image_array.shape[1],_3d_image_array.shape[2]))
   
   for var1 in range(3):
      for var2 in range(var1, 3):  # Only compute upper triangle (including diagonal)
         # Calculate gradients
         D1 = np.gradient(smoothed_image, axis=var1)
         D2 = np.gradient(D1, axis=var2)
         #print('D2 shape ',D2.shape)
         hessian[var1,var2] = hessian[var2,var1] = D2
         
   #print("hessian shape ", hessian.shape)

   # Compute the norm of the structure tensors for cval and Compute eigenvalues
   eig_vals = np.zeros((_3d_image_array.shape[0], _3d_image_array.shape[1], _3d_image_array.shape[2], 3))
   
   eig_vals = compute_eig_vals(hessian)
   
   #cvals = np.ones_like(_3d_image_array)*250

   #print('eig vals shape', eig_vals.shape, eig_vals[100,100,100])
   
   return eig_vals

def compute_vesselness(eigvals, alpha, beta):
   """Compute vesselness measure from Hessian elements."""
   # Extract Hessian elements
   lambdas1 = eigvals[:,:,:,0]
   lambdas2 = eigvals[:,:,:,1]
   lambdas3 = eigvals[:,:,:,2]
   #print("lambdas shapes ", lambdas1.shape)

   # Compute vesselness measure
   Ra = divide_nonzero(np.abs(lambdas2), np.abs(lambdas3))
   #print("Ra shapes ", Ra.shape)
   Rb = divide_nonzero(np.abs(lambdas1), np.sqrt(np.abs(np.multiply(lambdas2, lambdas3))))
   #print("Rb shapes ", Rb.shape)
   S = np.sqrt(np.square(lambdas1) + np.square(lambdas2) + np.square(lambdas3))
   #print("S : ", S.max())
   
   gamma = S.max() / 2
   if gamma == 0:
         gamma = 1  # If s == 0 everywhere, gamma doesn't matter.

   vesselness = (1 - np.exp(-Ra**2 / (2 * alpha**2))) * np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * gamma**2)))
   #print("vesselness shapes ", vesselness.shape)
   return vesselness
