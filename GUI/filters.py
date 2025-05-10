import math
import time
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import scipy.ndimage as ndi
import nibabel as nib
from skimage import filters
from skimage.transform import resize
from skimage.restoration import estimate_sigma

min_voxel_size = None

def set_min_voxel_size(mm):
   global min_voxel_size
   min_voxel_size = mm

def delete_temp():
   """
   Deletes all files in the "temp" folder.
   """
   
   global nii_3d_image, nii_3d_image_original
   
   folder_path = "temp"
   nii_3d_image = []
   nii_3d_image_original = []
   
   if os.path.exists(folder_path):
      for filename in os.listdir(folder_path):
         file_path = os.path.join(folder_path, filename)
         try:
               if os.path.isfile(file_path):
                  os.unlink(file_path)
               elif os.path.isdir(file_path):
                  shutil.rmtree(file_path)
         except Exception as e:
               print(f"Failed to delete {file_path}. Reason: {e}")
               print('='*100)
   else:
      print(f"The folder {folder_path} does not exist.")
      print('='*100)      


def isometric_voxels(image_file: nib.Nifti1Image) -> nib.Nifti1Image:
   """
   Resamples a NIfTI image to have isometric (equal-sized) voxels.

   If the input image already has isometric voxels, it is returned unchanged.  
   Otherwise, the image is resized so that all voxel dimensions match the smallest original voxel size,  
   and a new affine and header are adjusted accordingly.
   """
   
   header = image_file.header
   voxel_sizes = header.get_zooms()
   
   if all(v == voxel_sizes[0] for v in voxel_sizes): 
      print("Voxels are already Isometric")
      print('='*100)      
      return image_file
   else:
      shape = np.round(image_file.get_fdata().shape * (voxel_sizes / min_voxel_size)).astype(int)
      isometric_data = resize(image_file.get_fdata(), output_shape=shape, mode='constant')
      print("Image reshaped from ", image_file.get_fdata().shape, " to ", isometric_data.shape)
      print('='*100)      

      new_affine = image_file.affine.copy()
      new_affine[:3, :3] *= (voxel_sizes / min_voxel_size)[:, np.newaxis]
      new_nifti = nib.Nifti1Image(isometric_data, new_affine)
      new_nifti.header.set_zooms((min_voxel_size, min_voxel_size, min_voxel_size))
   
      return new_nifti

def mm2voxel(mm):
   num_voxels = mm / min_voxel_size
   return num_voxels

def voxel2mm(num_voxels):
   mm = num_voxels * min_voxel_size
   return mm

def divide_nonzero(array1, array2):
    """
    Divides two arrays. Returns zero when dividing by zero.
    """
    denominator = np.copy(array2)
    denominator[denominator == 0] = 1e-10
    return np.divide(array1, denominator)

def thresholding(data, threshold):
   """
   Applies a thresholding operation to a 3D image, setting values below the threshold to zero.
   """
   transformed_image = np.where(data > threshold, data, 0)
   return transformed_image
   
def thresholding2d(data, threshold):
   """
   Applies a thresholding operation to a 2D image, setting values below the threshold to zero, 
   and saves the resulting image as a JPEG file, limited to 30 frames per second.
   """
   t = time.time()
   if t - thresholding2d.last_time > 1/30:
      transformed_image = np.where(data > threshold, data, 0)
      plt.imsave("temp/plot.jpeg", transformed_image, cmap = 'gray')
      thresholding2d.last_time = t
thresholding2d.last_time = 0

def gaussian_preview(data, intensity):
   """
   Applies a Gaussian filter to the input data to preview the effect of different intensity values.
   If a root window is provided, the result is saved to a temporary file asynchronously; otherwise, it's saved directly.
   Limits the preview to 30 frames per second.
   """
   
   t = time.time()
   if t - gaussian_preview.last_time > 1/30:
      data = ndi.gaussian_filter(data, intensity, mode='constant')
      plt.imsave("temp/plot.jpeg", data, cmap='gray')
      gaussian_preview.last_time = t

gaussian_preview.last_time = 0
   

def gaussian3d(data3D, intensity):
   """
   Applies a 3D Gaussian filter to the input data, with the filter's kernel size determined by the intensity value.
   Ensures that the kernel size is odd.
   """
   kernel_size = max(1, math.trunc(intensity))
   if kernel_size % 2 == 0:
      kernel_size += 1  
   data = ndi.gaussian_filter(data3D, intensity/3, mode='constant')
   return data

def sci_frangi(image, scale_range=(1, 10), alpha=1, beta=0.5, steps=2, cval=1):
   """
   Applies the Frangi filter from scikit-image to the input image for vessel enhancement, 
   using a specified scale range, alpha, beta, and number of steps.
   """
   processed = filters.frangi(image, sigmas=np.arange(scale_range[0], scale_range[1], steps), alpha=alpha, beta=beta)
   print("sci-kit frangi applied")
   return processed

def get_sigma_and_snr(input_image: np.ndarray) -> dict:
   """
   Estimates noise level and signal-to-noise ratio (SNR) from a 3D MRI image.

   This function analyzes background noise from the image corners and signal intensity 
   from the center region. It performs Rician noise estimation and applies a correction 
   factor to return the adjusted noise sigma and the computed SNR.

   Parameters:
      input_image (np.ndarray): A 3D NumPy array representing the MRI image.

   Returns:
      tuple:
         actual_noise (float): Corrected Rician noise sigma.
         snr (float): Signal-to-noise ratio based on the center region intensity.
   """
   
   img_data = input_image
   corner_size = [int(dim * 0.05) for dim in img_data.shape]
   corners = [
      img_data[:corner_size[0], :corner_size[1], :corner_size[2]],   
      img_data[-corner_size[0]:, :corner_size[1], :corner_size[2]],  
      img_data[:corner_size[0], -corner_size[1]:, :corner_size[2]],  
      img_data[:corner_size[0], :corner_size[1], -corner_size[2]:],  
      img_data[-corner_size[0]:, -corner_size[1]:, :corner_size[2]], 
      img_data[-corner_size[0]:, :corner_size[1], -corner_size[2]:], 
      img_data[:corner_size[0], -corner_size[1]:, -corner_size[2]:], 
      img_data[-corner_size[0]:, -corner_size[1]:, -corner_size[2]:] 
   ]
   
   corner_means = [np.mean(corner) for corner in corners]
   median_intensity = np.median(corner_means)
   discard_threshold = 5 * median_intensity
   BG_corners = [corner for i, corner in enumerate(corners) if corner_means[i] <= discard_threshold]
   if len(BG_corners) == 0:
      print("No valid background corners found.")
      return None, None
   BG_noise_sample = np.concatenate([corner.flatten() for corner in BG_corners])
   
   center_size = [int(dim * 0.2) for dim in img_data.shape]
   RoI = img_data[
      img_data.shape[0]//2 - center_size[0]//2 : img_data.shape[0]//2 + center_size[0]//2,
      img_data.shape[1]//2 - center_size[1]//2 : img_data.shape[1]//2 + center_size[1]//2,
      img_data.shape[2]//2 - center_size[2]//2 : img_data.shape[2]//2 + center_size[2]//2
   ]
   mean_signal_intensity_RoI = np.mean(RoI)
   BG_noise_sigma = estimate_sigma(BG_noise_sample, average_sigmas=True)
   if np.isnan(BG_noise_sigma):
      BG_noise_sigma = 0.0
   snr = mean_signal_intensity_RoI / BG_noise_sigma
   actual_noise = BG_noise_sigma*2*0.875
   
   return actual_noise, snr

def my_frangi_filter(image, sigmas=[1], alpha=1, beta=0.5, black_vessels=True):
   """
   Applies the Frangi filter to an image for vesselness detection across multiple scales.

   The function normalizes the input image, optionally inverts it for black vessels, 
   and iterates over specified sigma values to compute vesselness at different scales.
   The final vesselness result is the averaged output from all scales.
   """

   image = image.astype(np.float64)
   image /= np.max(image)
   if black_vessels:
        image = -image
   vesselness = np.zeros_like(image)
   
   for sigma in sigmas:
      print('Current scale:', sigma)
      eigenvalues = compute_hessian_return_eigvals(image, sigma=sigma)
      output = compute_vesselness(eigenvalues, alpha, beta).astype(np.float64)
      vesselness += output
   
   print("Frangi filter applied.")
   return vesselness/np.max(vesselness)

def compute_eig_vals(hessian):
   """
   Computes the eigenvalues of a Hessian matrix for a 3D image.

   The function calculates the eigenvalues of a given Hessian matrix and sorts them in descending order.
   """
   
   eigvals = np.linalg.eigvals(hessian.transpose(2, 3, 4, 0, 1))
   sorted_eigvals = -np.sort(-eigvals, axis=-1)

   return sorted_eigvals

def compute_cvals(hessian):
   """
   Computes the norm of the structure tensors for use in vesselness calculations.

   This function calculates the norm of the structure tensors, which is used to evaluate the sharpness and contrast of the vessel structures.
   """
   
   norm = np.linalg.norm(hessian, axis=(0, 1)) / 2
   
   return norm

def compute_hessian_return_eigvals(_3d_image_array, sigma=1):
   """
   Computes the Hessian matrix and its eigenvalues for a 3D image.

   The function applies Gaussian smoothing and gradient computation to obtain the Hessian matrix for a 3D image. 
   It then computes and returns the eigenvalues of the Hessian matrix, which are used in vesselness calculations.
   """

   float_dtype = np.float64
   _3d_image_array = _3d_image_array.astype(float_dtype, copy=False)
   if np.isscalar(sigma):
      sigma = (sigma,) * 3 

   smoothed_image = ndi.gaussian_filter(_3d_image_array, sigma=sigma, mode='nearest')
   hessian = np.zeros((3,3,_3d_image_array.shape[0],_3d_image_array.shape[1],_3d_image_array.shape[2]))
   
   for var1 in range(3):
      for var2 in range(var1, 3):  
         D1 = np.gradient(smoothed_image, axis=var1)
         D2 = np.gradient(D1, axis=var2)
         hessian[var1,var2] = hessian[var2,var1] = D2
         
   eig_vals = np.zeros((_3d_image_array.shape[0], _3d_image_array.shape[1], _3d_image_array.shape[2], 3))
   eig_vals = compute_eig_vals(hessian)
   
   return eig_vals

def compute_vesselness(eigvals, alpha, beta):
   """
   Computes the vesselness measure from the eigenvalues of the Hessian matrix.

   This function calculates a vesselness score based on the eigenvalues of the Hessian, using specific parameters (alpha, beta, gamma) to refine the vesselness measure.
   """

   lambdas1 = eigvals[:,:,:,0]
   lambdas2 = eigvals[:,:,:,1]
   lambdas3 = eigvals[:,:,:,2]

   Ra = divide_nonzero(np.abs(lambdas2), np.abs(lambdas3))
   Rb = divide_nonzero(np.abs(lambdas1), np.sqrt(np.abs(np.multiply(lambdas2, lambdas3))))
   S = np.sqrt(np.square(lambdas1) + np.square(lambdas2) + np.square(lambdas3))
   
   gamma = S.max() / 2
   if gamma == 0:
         gamma = 1  

   vesselness = (1 - np.exp(-Ra**2 / (2 * alpha**2))) * np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * gamma**2)))
   
   return vesselness
