import math
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

def black_vessels(header):   
   # Get relevant metadata
   tr = header.get('repetition_time', None)
   te = header.get('echo_time', None)
   ti = header.get('inversion_time', None)
   description = str(header.get('descrip', '')).lower()
   
   black_vessels = True

   # Classification logic
   if tr is not None and te is not None:
      if 'flair' in description or (ti is not None and ti > 1800):  # Assuming TI > 1800 ms indicates FLAIR
         print('FLAIR')  # FLAIR
         black_vessels = True
      elif tr < 1000 and te < 30:  # Typical T1 parameters
         print('T1w')  # T1w
         black_vessels = True
      elif tr > 2000 and te > 100:  # Typical T2 parameters
         print('T2w')  # T2w
         black_vessels = False
      else:
         print('Unknown')  # Unknown
   else:
      print('Metadata missing')  # Metadata missing
      
   return black_vessels

def median_filter(image, kernel_size=(3, 3, 3)):
   """
   Apply a median filter to a 3D image array.

   Parameters:
   - image: 3D numpy array representing the input image.
   - kernel_size: Tuple specifying the size of the kernel in each dimension (e.g., (3, 3, 3)).

   Returns:
   - filtered_image: 3D numpy array containing the filtered image.
   """
   # Get image shape
   depth, height, width = image.shape

   # Initialize filtered image
   filtered_image = np.zeros_like(image)

   # Pad the image to handle borders
   padded_image = np.pad(image, [(k // 2, k // 2) for k in kernel_size], mode='constant')

   # Iterate over each voxel in the image
   for d in range(depth):
      for h in range(height):
         for w in range(width):
               # Extract the neighborhood of the current voxel
               neighborhood = padded_image[d:d+kernel_size[0], h:h+kernel_size[1], w:w+kernel_size[2]]

               # Compute the median of the neighborhood
               median_value = np.median(neighborhood)

               # Assign the median value to the corresponding voxel in the filtered image
               filtered_image[d, h, w] = median_value
   return filtered_image

def delete_temp():
   global nii_3d_image, nii_3d_image_original
   # Delete the contents of the temp folder
   folder_path = "temp"

   nii_3d_image = []
   nii_3d_image_original = []
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


def isometric_voxels(image_file: nib.Nifti1Image) -> nib.Nifti1Image:
   
   header = image_file.header
   # Get voxel sizes
   voxel_sizes = header.get_zooms()
   min_voxel_size = min(voxel_sizes)
   
   # Resample the image data to ensure isometric voxels
   if all(v == voxel_sizes[0] for v in voxel_sizes): #no need to resize
      print("Voxels are already Isometric")
      return image_file
   else:
      shape = np.round(image_file.get_fdata().shape * (voxel_sizes / min_voxel_size)).astype(int)
      isometric_data = resize(image_file.get_fdata(), output_shape=shape, mode='constant')
      print("Image reshaped from ", image_file.get_fdata().shape, " to ", isometric_data.shape)

      # Create new affine with the new voxel size
      new_affine = image_file.affine.copy()
      new_affine[:3, :3] *= (voxel_sizes / min_voxel_size)[:, np.newaxis]

      # Create and return the new NIfTI image with updated zooms
      new_nifti = nib.Nifti1Image(isometric_data, new_affine)
      new_nifti.header.set_zooms((min_voxel_size, min_voxel_size, min_voxel_size))
   
      return new_nifti


def mm2voxel(mm):
   num_voxels = mm / min_voxel_size
   #print("for a voxel size of ",voxel_size," cubic mm, ",mm, "mm's of diameter is ",num_voxels," voxels")
   return num_voxels #returns the diameter

def voxel2mm(num_voxels):
   mm = num_voxels * min_voxel_size
   #print("for a voxel size of ",voxel_size," cubic mm, ",num_voxels, " voxels is equivalent to ",mm," mm")
   return mm  # returns the diameter in mm

def oldcalculate_noise(input_image: np.ndarray):
   img_data = input_image
   # Ajustar el tamaño del filtro según las dimensiones de la imagen
   filter_size = max(3, min(img_data.shape) // 100)  # Ejemplo dinámico: un tamaño base de 3, ajustado por la escala de la imagen
   
   # Aplicar el filtro de mediana
   smooth_image = ndi.median_filter(img_data, size=filter_size)
   
   # Resto del cálculo es igual
   residual_noise = img_data - smooth_image
   
   corner_size = [int(dim * 0.05) for dim in img_data.shape]
   
   corners = [
   residual_noise[:corner_size[0], :corner_size[1], :corner_size[2]],   # (0,0,0)
   residual_noise[-corner_size[0]:, :corner_size[1], :corner_size[2]],  # (end,0,0)
   residual_noise[:corner_size[0], -corner_size[1]:, :corner_size[2]],  # (0,end,0)
   residual_noise[:corner_size[0], :corner_size[1], -corner_size[2]:],  # (0,0,end)
   residual_noise[-corner_size[0]:, -corner_size[1]:, :corner_size[2]], # (end,end,0)
   residual_noise[-corner_size[0]:, :corner_size[1], -corner_size[2]:], # (end,0,end)
   residual_noise[:corner_size[0], -corner_size[1]:, -corner_size[2]:], # (0,end,end)
   residual_noise[-corner_size[0]:, -corner_size[1]:, -corner_size[2]:], # (end,end,end)
   ]
   
   # Combinar todas las muestras de las esquinas en una muestra grande
   large_sample = np.concatenate([corner.flatten() for corner in corners])
   
   std_deviation_noise = np.std(large_sample)   

   scaled_sigma = std_deviation_noise
   
   return scaled_sigma


def calculate_noise(input_image: np.ndarray):
   img_data = input_image
   # Ajustar el tamaño del filtro según las dimensiones de la imagen
   
   img_data = input_image
   #downsample the image to blur
   #toblur = resize(input_image, (input_image.shape[0] // 2, input_image.shape[1] // 2, input_image.shape[2] // 2))
   #blurred_image = ndi.median_filter(toblur, size=3)
   #max_intensity = blurred_image.max()
   #print('max intensity ', max_intensity)
   
   # Tamaño de las esquinas: 5% de las dimensiones de la imagen
   corner_size = [int(dim * 0.05) for dim in img_data.shape]
   
   # Seleccionar las esquinas
   corners = [
       img_data[:corner_size[0], :corner_size[1], :corner_size[2]],   # (0,0,0)
       img_data[-corner_size[0]:, :corner_size[1], :corner_size[2]],  # (end,0,0)
       img_data[:corner_size[0], -corner_size[1]:, :corner_size[2]],  # (0,end,0)
       img_data[:corner_size[0], :corner_size[1], -corner_size[2]:],  # (0,0,end)
       img_data[-corner_size[0]:, -corner_size[1]:, :corner_size[2]], # (end,end,0)
       img_data[-corner_size[0]:, :corner_size[1], -corner_size[2]:], # (end,0,end)
       img_data[:corner_size[0], -corner_size[1]:, -corner_size[2]:], # (0,end,end)
       img_data[-corner_size[0]:, -corner_size[1]:, -corner_size[2]:] # (end,end,end)
   ]
   
   # Aplicar el filtro de mediana solo a las esquinas
   filtered_corners = [ndi.median_filter(corner, size=3) for corner in corners]
   
   # Calcular el ruido residual (restar el filtro mediana aplicado en las esquinas)
   residual_noise_corners = [corner - filtered_corner for corner, filtered_corner in zip(corners, filtered_corners)]
   
   # Combinar todas las muestras de las esquinas en una muestra grande
   large_sample = np.concatenate([corner.flatten() for corner in residual_noise_corners])
   
   # Calcular la desviación estándar del ruido
   std_deviation_noise = np.std(large_sample)

   scaled_sigma = std_deviation_noise
   
   return scaled_sigma*2



def all_noise_measurements(input_image: np.ndarray) -> dict:
   """
   Estimate noise in an MRI image using various methods, considering only background corners.
   
   :param input_image: 3D MRI image data
   :return: Dictionary with noise estimation results.
   """
   img_data = input_image

   # 1. Background Region Method (ROI from image corners)
   # Select corners (5% of the dimensions of the image)
   corner_size = [int(dim * 0.05) for dim in img_data.shape]
   corners = [
      img_data[:corner_size[0], :corner_size[1], :corner_size[2]],   # (0,0,0)
      img_data[-corner_size[0]:, :corner_size[1], :corner_size[2]],  # (end,0,0)
      img_data[:corner_size[0], -corner_size[1]:, :corner_size[2]],  # (0,end,0)
      img_data[:corner_size[0], :corner_size[1], -corner_size[2]:],  # (0,0,end)
      img_data[-corner_size[0]:, -corner_size[1]:, :corner_size[2]], # (end,end,0)
      img_data[-corner_size[0]:, :corner_size[1], -corner_size[2]:], # (end,0,end)
      img_data[:corner_size[0], -corner_size[1]:, -corner_size[2]:], # (0,end,end)
      img_data[-corner_size[0]:, -corner_size[1]:, -corner_size[2]:] # (end,end,end)
   ]

   # Compute the mean intensity for each corner
   corner_means = [np.mean(corner) for corner in corners]

   # Filter out corners with mean intensities significantly higher than the median of the other corners, the noise is trusted to be uniform in the hole image, if one of the corers is significantly higher, its discarded due to the fact that they might have important tissue
   median_intensity = np.median(corner_means)
   threshold = 5 * median_intensity
   background_corners = [corner for i, corner in enumerate(corners) if corner_means[i] <= threshold]

   # Flatten and combine the remaining valid background corners
   if len(background_corners) == 0:
      print("No valid background corners found.")
      return None, None
   
   large_sample = np.concatenate([corner.flatten() for corner in background_corners])

   # 2. Select Center Region of interest (ROI) (20% of the image dimensions)
   center_size = [int(dim * 0.2) for dim in img_data.shape]
   center_region = img_data[
      img_data.shape[0]//2 - center_size[0]//2 : img_data.shape[0]//2 + center_size[0]//2,
      img_data.shape[1]//2 - center_size[1]//2 : img_data.shape[1]//2 + center_size[1]//2,
      img_data.shape[2]//2 - center_size[2]//2 : img_data.shape[2]//2 + center_size[2]//2
   ]
   
   # Compute mean intensity in the center region
   mean_signal_intensity_center = np.mean(center_region)
   
   # 3. Rician Noise Estimation (using skimage)
   rician_noise_sigma = estimate_sigma(large_sample, average_sigmas=True)
   
   # Avoid NaN or very small values for Rician noise sigma
   if np.isnan(rician_noise_sigma):
      rician_noise_sigma = 0.0

   # SNR calculation based on mean intensity of the center region
   snr = mean_signal_intensity_center / rician_noise_sigma
   
   actual_noise = rician_noise_sigma*2*0.875
   
   return actual_noise, snr



def new_calculate_noise(input_image: np.ndarray):
   img_data = input_image
   #downsample the image to blur
   toblur = resize(input_image, (input_image.shape[0] // 2, input_image.shape[1] // 2, input_image.shape[2] // 2))
   blurred_image = ndi.median_filter(toblur, size=int(mm2voxel(4)))
   max_intensity = blurred_image.max()
   print('max intensity ', max_intensity)
   
   # Tamaño de las esquinas: 5% de las dimensiones de la imagen
   corner_size = [int(dim * 0.05) for dim in img_data.shape]
   
   # Seleccionar las esquinas
   corners = [
       img_data[:corner_size[0], :corner_size[1], :corner_size[2]],   # (0,0,0)
       img_data[-corner_size[0]:, :corner_size[1], :corner_size[2]],  # (end,0,0)
       img_data[:corner_size[0], -corner_size[1]:, :corner_size[2]],  # (0,end,0)
       img_data[:corner_size[0], :corner_size[1], -corner_size[2]:],  # (0,0,end)
       img_data[-corner_size[0]:, -corner_size[1]:, :corner_size[2]], # (end,end,0)
       img_data[-corner_size[0]:, :corner_size[1], -corner_size[2]:], # (end,0,end)
       img_data[:corner_size[0], -corner_size[1]:, -corner_size[2]:], # (0,end,end)
       img_data[-corner_size[0]:, -corner_size[1]:, -corner_size[2]:] # (end,end,end)
   ]
   
   # Combinar todas las muestras de las esquinas en una muestra grande
   large_sample = np.concatenate([corner.flatten() for corner in corners])
   
   # Calcular la desviación estándar del ruido
   std_deviation_noise = np.std(large_sample)

   scaled_sigma = std_deviation_noise / max_intensity
   #scaled_sigma = std_deviation_noise / np.median(large_sample)
   
   return scaled_sigma

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
