import torch
import time
import numpy as np
import scipy.ndimage as ndi
from concurrent.futures import ThreadPoolExecutor
import sounds
import torch.nn.functional as F
import filters


def apply_gaussian_blur_3d(image: torch.Tensor, sigma: float) -> torch.Tensor:
   """
   Apply 3D Gaussian blur to the image using scipy's gaussian_filter.
   """
   # Ensure image is on CPU for scipy operations
   image = image.cpu().numpy()
   
   # Apply Gaussian filter
   blurred_image = ndi.gaussian_filter(image, sigma=sigma, mode='constant')
   
   # Convert back to torch tensor
   blurred_image = torch.tensor(blurred_image, dtype=torch.float32, device=image.device)
   return blurred_image

def divide_nonzero(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
   """
   Divide elements of a by b, handling division by zero.
   """
   result = torch.where(b != 0, a / b, torch.zeros_like(a))
   return result

def compute_vesselness(eigvals: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
   """
   Compute vesselness measure from Hessian eigenvalues.
   """
   lambdas1 = eigvals[:,:,:,0]
   lambdas2 = eigvals[:,:,:,1]
   lambdas3 = eigvals[:,:,:,2]

   Ra = divide_nonzero(torch.abs(lambdas2), torch.abs(lambdas3))
   Rb = divide_nonzero(torch.abs(lambdas1), torch.sqrt(torch.abs(lambdas2 * lambdas3)))
   S = torch.sqrt(torch.square(lambdas1) + torch.square(lambdas2) + torch.square(lambdas3)).real

   gamma = S.max() / 2
   gamma = torch.where(gamma == 0, torch.tensor(1.0, dtype=lambdas1.dtype), gamma)

   vesselness = (1 - torch.exp(-torch.square(Ra) / (2 * alpha**2))) * \
               torch.exp(-torch.square(Rb) / (2 * beta**2)) * \
               (1 - torch.exp(-torch.square(S) / (2 * gamma**2)))

   return vesselness

def compute_eig_vals(hessian: torch.Tensor) -> torch.Tensor:
   """
   Compute eigenvalues of the Hessian matrix.
   """
   # Convert the tensor to a NumPy array
   hessian_np = hessian.cpu().numpy()
   
   # Compute eigenvalues
   eigvals = np.linalg.eigvals(hessian_np.transpose(2, 3, 4, 0, 1))  # Move the 3x3 matrices to the last dimensions

   # Sort eigenvalues in descending order
   sorted_eigvals = np.sort(eigvals, axis=-1)
   return torch.from_numpy(sorted_eigvals)

def compute_hessian_return_eigvals(image: torch.Tensor, sigma: float) -> torch.Tensor:
   """
   Compute the Hessian matrix and its eigenvalues using convolutions.
   """

   smoothed_image = apply_gaussian_blur_3d(image, sigma=sigma)
   
   # Compute gradients
   gradients = torch.gradient(smoothed_image)
   gradients = torch.stack(gradients)

   hessian = torch.zeros((3, 3, *smoothed_image.shape))

   for var1 in range(3):
      for var2 in range(var1, 3):
         D1 = torch.gradient(smoothed_image, dim=var1)[0]
         D2 = torch.gradient(D1, dim=var2)[0]
         hessian[var1, var2] = hessian[var2, var1] = D2

   eig_vals = compute_eig_vals(hessian)
   
   return eig_vals

'''
def testing_process_scale(tensor_image: torch.Tensor, sigma: float, alpha: float, beta: float, mask: np.ndarray, shell: np.ndarray):
   global data_list
   text = f'Scale {sigma.item():.2f} finished'
   eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=sigma)
   output = compute_vesselness(eigenvalues, alpha, beta).real
   
   mask1 = mask.copy()
   mask2 = mask.copy()
   mask3 = mask.copy()
   mask4 = mask.copy()
   mask5 = mask.copy()
   mask6 = mask.copy()
   mask7 = mask.copy()
   mask1[75:,:,:] = 0
   mask2[:75,:,:] = 0
   mask2[125:,:,:] = 0
   mask3[:125,:,:] = 0
   mask3[175:,:,:] = 0
   mask4[:175,:,:] = 0
   mask4[225:,:,:] = 0
   mask5[:225,:,:] = 0
   mask5[275:,:,:] = 0
   mask6[:275,:,:] = 0
   mask6[325:,:,:] = 0
   mask7[:325,:,:] = 0
   shell1 = shell.copy()
   shell2 = shell.copy()
   shell3 = shell.copy()
   shell4 = shell.copy()
   shell5 = shell.copy()
   shell6 = shell.copy()
   shell7 = shell.copy()
   shell1[75:,:,:] = 0
   shell2[:75,:,:] = 0
   shell2[125:,:,:] = 0
   shell3[:125,:,:] = 0
   shell3[175:,:,:] = 0
   shell4[:175,:,:] = 0
   shell4[225:,:,:] = 0
   shell5[:225,:,:] = 0
   shell5[275:,:,:] = 0
   shell6[:275,:,:] = 0
   shell6[325:,:,:] = 0
   shell7[:325,:,:] = 0
   avg_intensity_B_1 = torch.mean(output[shell1 ==1])
   avg_intensity_B_2 = torch.mean(output[shell2 ==1])
   avg_intensity_B_3 = torch.mean(output[shell3 ==1])
   avg_intensity_B_4 = torch.mean(output[shell4 ==1])
   avg_intensity_B_5 = torch.mean(output[shell5 ==1])
   avg_intensity_B_6 = torch.mean(output[shell6 ==1])
   avg_intensity_B_7 = torch.mean(output[shell7 ==1])
   
   mask1_tensor = torch.from_numpy(mask1).float()
   mask2_tensor = torch.from_numpy(mask2).float()
   mask3_tensor = torch.from_numpy(mask3).float()
   mask4_tensor = torch.from_numpy(mask4).float()
   mask5_tensor = torch.from_numpy(mask5).float()
   mask6_tensor = torch.from_numpy(mask6).float()
   mask7_tensor = torch.from_numpy(mask7).float()

   avg_intensity_F_1 = torch.sum(output * mask1_tensor) / torch.sum(mask1_tensor)
   avg_intensity_F_2 = torch.sum(output * mask2_tensor) / torch.sum(mask2_tensor)
   avg_intensity_F_3 = torch.sum(output * mask3_tensor) / torch.sum(mask3_tensor)
   avg_intensity_F_4 = torch.sum(output * mask4_tensor) / torch.sum(mask4_tensor)
   avg_intensity_F_5 = torch.sum(output * mask5_tensor) / torch.sum(mask5_tensor)
   avg_intensity_F_6 = torch.sum(output * mask6_tensor) / torch.sum(mask6_tensor)
   avg_intensity_F_7 = torch.sum(output * mask7_tensor) / torch.sum(mask7_tensor)


   
   avg_contrast_1 = max(0, avg_intensity_F_1 - avg_intensity_B_1)
   avg_contrast_2 = max(0, avg_intensity_F_2 - avg_intensity_B_2)
   avg_contrast_3 = max(0, avg_intensity_F_3 - avg_intensity_B_3)
   avg_contrast_4 = max(0, avg_intensity_F_4 - avg_intensity_B_4)
   avg_contrast_5 = max(0, avg_intensity_F_5 - avg_intensity_B_5)
   avg_contrast_6 = max(0, avg_intensity_F_6 - avg_intensity_B_6)
   avg_contrast_7 = max(0, avg_intensity_F_7 - avg_intensity_B_7)
   
   data_list.append([sigma.item(), avg_contrast_1.item(), avg_contrast_2.item(), avg_contrast_3.item(), avg_contrast_4.item(), avg_contrast_5.item(), avg_contrast_6.item(), avg_contrast_7.item()])
   print(text)
   return output
'''

data_list = []
def testing_process_scale(tensor_image: torch.Tensor, sigma: float, alpha: float, beta: float, mask: np.ndarray, shell: np.ndarray):
   global data_list
   text = f'Scale {sigma:.2f} finished'  # Remover .item() porque sigma es un float
   eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=sigma)
   output = compute_vesselness(eigenvalues, alpha, beta).real
   rangey = 8
   
   mask1 = mask.copy()
   mask2 = mask.copy()
   mask3 = mask.copy()
   mask4 = mask.copy()
   mask5 = mask.copy()
   mask6 = mask.copy()
   mask7 = mask.copy()
   mask1[75:,:,:] = 0
   mask2[:75,:,:] = 0
   mask2[125:,:,:] = 0
   mask3[:125,:,:] = 0
   mask3[175:,:,:] = 0
   mask4[:175,:,:] = 0
   mask4[225:,:,:] = 0
   mask5[:225,:,:] = 0
   mask5[275:,:,:] = 0
   mask6[:275,:,:] = 0
   mask6[325:,:,:] = 0
   mask7[:325,:,:] = 0
   shell1 = shell.copy()
   shell2 = shell.copy()
   shell3 = shell.copy()
   shell4 = shell.copy()
   shell5 = shell.copy()
   shell6 = shell.copy()
   shell7 = shell.copy()
   shell1[75:,:,:] = 0
   shell2[:75,:,:] = 0
   shell2[125:,:,:] = 0
   shell3[:125,:,:] = 0
   shell3[175:,:,:] = 0
   shell4[:175,:,:] = 0
   shell4[225:,:,:] = 0
   shell5[:225,:,:] = 0
   shell5[275:,:,:] = 0
   shell6[:275,:,:] = 0
   shell6[325:,:,:] = 0
   shell7[:325,:,:] = 0

   avg_intensity_B_1 = torch.mean(output[shell1 == 1])
   avg_intensity_B_2 = torch.mean(output[shell2 == 1])
   avg_intensity_B_3 = torch.mean(output[shell3 == 1])
   avg_intensity_B_4 = torch.mean(output[shell4 == 1])
   avg_intensity_B_5 = torch.mean(output[shell5 == 1])
   avg_intensity_B_6 = torch.mean(output[shell6 == 1])
   avg_intensity_B_7 = torch.mean(output[shell7 == 1])

   avg_intensity_F_1 = torch.mean(output[50, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])
   avg_intensity_F_2 = torch.mean(output[100, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])
   avg_intensity_F_3 = torch.mean(output[150, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])
   avg_intensity_F_4 = torch.mean(output[200, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])
   avg_intensity_F_5 = torch.mean(output[249, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])
   avg_intensity_F_6 = torch.mean(output[299, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])
   avg_intensity_F_7 = torch.mean(output[349, output.shape[1] // 2 - rangey:output.shape[1] // 2 + rangey, output.shape[2] // 2])

   avg_contrast_1 = max(0, (avg_intensity_F_1 - avg_intensity_B_1).item())
   avg_contrast_2 = max(0, (avg_intensity_F_2 - avg_intensity_B_2).item())
   avg_contrast_3 = max(0, (avg_intensity_F_3 - avg_intensity_B_3).item())
   avg_contrast_4 = max(0, (avg_intensity_F_4 - avg_intensity_B_4).item())
   avg_contrast_5 = max(0, (avg_intensity_F_5 - avg_intensity_B_5).item())
   avg_contrast_6 = max(0, (avg_intensity_F_6 - avg_intensity_B_6).item())
   avg_contrast_7 = max(0, (avg_intensity_F_7 - avg_intensity_B_7).item())

   # Ajustar aquí
   data_list.append([sigma.item(), avg_contrast_1, avg_contrast_2, avg_contrast_3, avg_contrast_4, avg_contrast_5, avg_contrast_6, avg_contrast_7])
   print(text)
   return output



def dataset_process_scale(tensor_image: torch.Tensor, sigma: float, alpha: float, beta: float, mask: np.ndarray, shell: np.ndarray) -> torch.Tensor:

   # Compute eigenvalues and vesselness
   eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=sigma)
   output = compute_vesselness(eigenvalues, alpha, beta).real

   # Reshape the output image to match the mask and shell shapes
   if mask.shape != output.shape:
      output_resampled = F.interpolate(output.unsqueeze(0).unsqueeze(0), size=mask.shape, mode='trilinear', align_corners=True)
      output_resampled = output_resampled.squeeze(0).squeeze(0)  # Remove added dimensions
      #print("Resampled output to shape:", output_resampled.shape)
   else:
      output_resampled = output
      #print('No output resampled')

   # Ensure the mask is a PyTorch tensor
   mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=output.device)
   shell_tensor = torch.as_tensor(shell, dtype=torch.float32, device=output.device)
   
   # Calculate the mean intensity for the mask
   avg_intensity_F = torch.mean(output_resampled[mask_tensor > 0.0])
   avg_intensity_B = torch.mean(output_resampled[shell_tensor > 0.5])
   
   avg_contrast = max(0, (avg_intensity_F - avg_intensity_B).item())

   # Append results to the global data list
   #data_list.append([filters.voxel2mm(sigma).item(), avg_contrast])
   
   print(f'Scale {sigma:.2f} (voxels) or {filters.voxel2mm(sigma):.2f} (mm) finished')

   return output, avg_contrast


def process_scale(tensor_image: torch.Tensor, sigma: float, alpha: float, beta: float):
   global data_list
   # Compute eigenvalues and vesselness
   eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=sigma)
   output = compute_vesselness(eigenvalues, alpha, beta).real
   print(f'Scale {sigma:.2f} finished')

   return output

def my_frangi_filter_parallel_training(input_image: np.ndarray, sigmas: list = [1], alpha: float = 1, beta: float = 0.5, black_vessels: bool = True, mask: np.ndarray = None, shell: np.ndarray = None) -> np.ndarray:
    global data_list
    print('Running Frangi filter in parallel')
    sounds.loading()

    # Start the timer
    start_time = time.time()

    tensor_image = torch.tensor(input_image, dtype=torch.float64)  # Ensure the image is float64
    
    if not black_vessels:
        tensor_image = -tensor_image
        print('Image inverted')

    vesselness = torch.zeros_like(tensor_image)

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(dataset_process_scale, tensor_image, sigma, alpha, beta, mask, shell) for sigma in sigmas]
        
        # Collect the results as they complete
        for future in futures:
            output, contrast = future.result()
            vesselness = torch.max(vesselness, output)

    vesselness /= vesselness.max()

    # Stop the timer
    end_time = time.time()
    duration = end_time - start_time
    sounds.success()
    print(f"Frangi filter applied in {duration:.4f} seconds.")

    # Sort data_list based on the first element (scale) of each triplet
    data_list.sort(key=lambda x: x[0])
    scales, avg_contrast = map(list, zip(*data_list))
    data_list = []
    print('scales =', scales)
    print('avg_contrast =', avg_contrast)

    vesselness_np = vesselness.cpu().numpy()
    return vesselness_np / vesselness_np.max()


def my_frangi_filter_parallel(input_image: np.ndarray, sigmas: list = [1], alpha: float = 1, beta: float = 0.5, black_vessels: bool = True) -> np.ndarray:
   global data_list
   tensor_image = torch.tensor(input_image, dtype=torch.float64)  # Ensure the image is float64
   #tensor_image /= tensor_image.max()  # Normalize the image
   print('Running Fragi filter in parallel')
   sounds.loading()
   if not black_vessels:
      tensor_image = -tensor_image
      print('image inverted')

   vesselness = torch.zeros_like(tensor_image)

   # Start the timer
   start_time = time.time()
   # Use ThreadPoolExecutor for parallel execution
   with ThreadPoolExecutor() as executor:
      futures = [executor.submit(process_scale, tensor_image, sigma, alpha, beta) for sigma in sigmas]
      
      # Collect the results as they complete
      for future in futures:
         output = future.result()
         vesselness = torch.max(vesselness, output)
   
   vesselness /= vesselness.max()
   # Stop the timer
   end_time = time.time()
   # Calculate the duration
   duration = end_time - start_time
   sounds.success()
   print(f"Frangi filter applied in {duration:.4f} seconds.")
   # First, sort the data_list based on the first element (scale) of each triplet
   
   vesselness_np = vesselness.cpu().numpy()
   return vesselness_np / vesselness_np.max()