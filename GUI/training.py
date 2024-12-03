from scipy.interpolate import make_interp_spline
import torch
import tensor_frangi
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from scipy.ndimage import zoom
from scipy.ndimage import distance_transform_edt
import nibabel as nib
import numpy as np
import sounds
import filters
import os
import gc


def resample_nifti_to_voxel(nifti_image: nib.Nifti1Image, target_voxel_size=(1, 1, 1)) -> nib.Nifti1Image:
   """
   Resamples a NIfTI image to the specified target voxel dimensions, creating a new affine and header.

   Parameters:
   - nifti_image: nibabel.Nifti1Image, the input NIfTI image to be resampled.
   - target_voxel_size: tuple of floats, the desired voxel size in mm (e.g., (1, 1, 1) for 1x1x1 mm).

   Returns:
   - A resampled NIfTI image with the new voxel size.
   """
   # Get original voxel size from the image header
   original_voxel_size = nifti_image.header.get_zooms()[:3]

   # Calculate the scaling factors along each dimension
   scale_factors = [orig / target for orig, target in zip(original_voxel_size, target_voxel_size)]

   # Get the original image data
   img_data = nifti_image.get_fdata()

   # Resample the image data
   resampled_data = zoom(img_data, zoom=scale_factors, order=1)  # Using order=1 for bilinear interpolation

   # Create a new affine transformation matrix for the target voxel size
   new_affine = np.eye(4)
   for i in range(3):
      new_affine[i, i] = target_voxel_size[i]

   # Create a new header with the target voxel size
   new_header = nib.Nifti1Header()
   new_header.set_data_shape(resampled_data.shape)
   new_header.set_zooms(target_voxel_size)

   # Create a new NIfTI image with the resampled data, new affine, and new header
   resampled_nifti = nib.Nifti1Image(resampled_data, new_affine, new_header)

   return resampled_nifti



def generate_noise(image: nib.Nifti1Image, noise_start=None, noise_end=None) -> list:
   voxelsize = image.header.get_zooms()[0]   
   def random_resample(image: nib.Nifti1Image) -> nib.Nifti1Image:
      #rand value between 0.5 and 3 which are the common voxel sizes in MRIs
      #rand_size = 0.5
      rand_size = np.random.uniform(voxelsize, 1.8)
      #save 2 digits only
      rand_size = round(rand_size, 2)
      resampled_image = resample_nifti_to_voxel(image, target_voxel_size=(rand_size,rand_size,rand_size))
      return resampled_image
   
   # Function to add Rician noise to an image
   def add_rician_noise(image: np.ndarray, sigma: float) -> np.ndarray:
      noise = np.random.normal(loc=0, scale=sigma, size=image.shape)
      noisy_image = np.sqrt(image**2 + noise**2)
      return noisy_image

   # Function to create a noisy image with Rician noise
   def create_noisy_image(base_image: np.ndarray, sigma: float) -> np.ndarray:
      noisy_image = add_rician_noise(base_image, sigma)
      return noisy_image
   
   if noise_start is None:
      noise_sigmas = np.linspace(0, image.get_fdata().max()*0.09, 21)
   else:
      noise_sigmas = np.linspace(noise_start, noise_end, 11)
   
   noisy_images = []
   
   # Loop through noise levels, create noisy images, and store them as NIfTI files
   for sigma in noise_sigmas:
      random_resampled_image = random_resample(image)
      #random_resampled_image = image
      data = random_resampled_image.get_fdata()
      rand_noise = np.random.uniform(0, image.get_fdata().max()*0.09)
      noisy_image_data = create_noisy_image(data, rand_noise)
      
      # Create a new NIfTI image with the same affine and header as the input image
      noisy_nifti = nib.Nifti1Image(noisy_image_data, affine=random_resampled_image.affine)
      noisy_nifti.header.set_zooms(random_resampled_image.header.get_zooms())
      
      # Append the new NIfTI image to the list
      noisy_images.append(noisy_nifti)
   
   return noisy_images


def create_shell(mask: np.ndarray) -> np.ndarray:
   # Get the original image (mask or object)
   tomask = mask
   
   # Compute the distance transform of the inverted mask (background)
   distance_transform = distance_transform_edt(tomask == 0)  # Now EDT of background
   
   outer_threshold = filters.mm2voxel(1.5)
   inner_threshold = filters.mm2voxel(0.1)
   
   # Create the outer shell: Select voxels within the outer threshold of the object's boundary
   shell = ((distance_transform > inner_threshold) & (distance_transform <= outer_threshold)).astype(float)
   
   return shell

def process_scale(noisy_image: torch.Tensor, scale: float, alpha: float, beta: float, mask: np.ndarray, shell: np.ndarray):
   # Runs filtering for a single scale
   filtered_output, avg_contrast = tensor_frangi.dataset_process_scale(noisy_image, scale, alpha, beta, mask=mask, shell=shell)
   return avg_contrast

def process_image_get_contrasts(noisy_image: torch.Tensor, scales: list, alpha: float, beta: float, mask: np.ndarray, shell: np.ndarray):
   contrasts = []
   
   for scale in scales:
      # Process each scale and collect its result
      avg_contrast = process_scale(noisy_image, scale, alpha, beta, mask, shell)
      contrasts.append(avg_contrast)
      
      # Free memory after each scale is processed
      del avg_contrast
      gc.collect()

   return contrasts

def interpolate_and_find_best_scale(scales, contrasts, num_points=300):
   best_scale = None

   # Interpolation for smooth lines with increased resolution (300 points)
   scales_new = np.linspace(min(scales), max(scales), num_points)  # New scale with higher resolution
   f = make_interp_spline(scales, contrasts, k=3)(scales_new)

   # Find the x value (scale) where each smooth function reaches its maximum
   best_scale = scales_new[np.argmax(f)]
   
   return best_scale

def save_results(image_name, noise_level, snr, voxel_size, best_scale):
    # Open the CSV file in append mode ('a')
    with open('results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Check if the file is empty to write the header
        if file.tell() == 0:
            writer.writerow(['Image Name', 'STD sigma', 'SNR', 'Voxel Size (mm)', 'Best Scale (mm)'])
        
        # Write the data
        writer.writerow([image_name, noise_level, snr, voxel_size, best_scale])
#mri_images is a
#remember to give the scales in voxels!!!
def automate_mri_analysis(mri_images: list[nib.Nifti1Image], mask: nib.Nifti1Image, voxel_scales):

   print("Automating MRI analysis...")
   mask = mask.get_fdata()
   shell = create_shell(mask)
   
   for mri_image in mri_images:
      #get the name of the file
      Image_name = mri_image.get_filename()
      min_voxel_size = min(mri_image.header.get_zooms()[0])
      filters.set_min_voxel_size(min_voxel_size)
      Image_name = os.path.basename(Image_name)
      print('Processing ', Image_name)
      mri_image = filters.isometric_voxels(mri_image)
                  
      noise_sigma, snr = filters.all_noise_measurements(mri_image.get_fdata())
      noisy_image_tensor = torch.from_numpy(mri_image.get_fdata())
      contrasts = process_image_get_contrasts(noisy_image_tensor, voxel_scales, 0.2, 0.8, mask, shell)
      voxel_best_scale = interpolate_and_find_best_scale(voxel_scales, contrasts)
      #save the results in a csv, the data to store is: the name of the file, the noise level, the SNR, the best scale and the minimum voxel size
      min_voxel_size = min(mri_image.header.get_zooms())
      filters.set_min_voxel_size(min_voxel_size)
      scale_mm = filters.voxel2mm(voxel_best_scale)
      save_results(Image_name, noise_sigma, snr, min_voxel_size, scale_mm)
      print("MRI analysis for ", Image_name, " completed and saved.")
      sounds.success()
