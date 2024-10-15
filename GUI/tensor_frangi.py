import torch
import time
import numpy as np
import scipy.ndimage as ndi
from concurrent.futures import ThreadPoolExecutor
from pygame import mixer


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

def compute_hessian_return_eigvals(image: torch.Tensor, sigma: float = 1) -> torch.Tensor:
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

data_list = []
def ptestin_rocess_scale(tensor_image: torch.Tensor, sigma: float, alpha: float, beta: float, mask: np.ndarray):
   global data_list
   # Compute eigenvalues and vesselness
   eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=sigma)
   output = compute_vesselness(eigenvalues, alpha, beta).real
   print(f'Scale {sigma:.2f} finished')

   return output

def process_scale(tensor_image: torch.Tensor, sigma: float, alpha: float, beta: float, mask: np.ndarray):
   global data_list
   # Compute eigenvalues and vesselness
   eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=sigma)
   output = compute_vesselness(eigenvalues, alpha, beta).real
   print(f'Scale {sigma:.2f} finished')

   return output


def my_frangi_filter_parallel(input_image: np.ndarray, sigmas: list = [1], alpha: float = 1, beta: float = 0.5, black_vessels: bool = True, mask: np.ndarray = None) -> np.ndarray:
   global data_list
   tensor_image = torch.tensor(input_image, dtype=torch.float64)  # Ensure the image is float64
   tensor_image /= tensor_image.max()  # Normalize the image
   print('Running Fragi filter in parallel')
   mixer.init()
   mixer.music.load('sounds/loading.mp3')
   mixer.music.play()
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
   mixer.music.load('sounds/done.mp3')
   mixer.music.play()
   print(f"Frangi filter applied in {duration:.4f} seconds.")
   # First, sort the data_list based on the first element (scale) of each triplet
   '''
   data_list.sort(key=lambda x: x[0])
   scales, avg_F_1 = map(list, zip(*data_list))
   data_list = []
   print('scales =',scales)
   print('avg_F_1 =',avg_F_1)
   '''
   
   '''
   data_list.sort(key=lambda x: x[0])
   scales, avg_B_1, avg_B_2, avg_B_3, avg_B_4, avg_B_5, avg_B_6, avg_B_7, avg_F_1, avg_F_2, avg_F_3, avg_F_4, avg_F_5, avg_F_6, avg_F_7,  = map(list, zip(*data_list))
   data_list = []
   print('scales =',scales)
   print('avg_B_1 =',avg_B_1)
   print('avg_B_2 =',avg_B_2)
   print('avg_B_3 =',avg_B_3)
   print('avg_B_4 =',avg_B_4)
   print('avg_B_5 =',avg_B_5)
   print('avg_B_6 =',avg_B_6)
   print('avg_B_7 =',avg_B_7)
   
   print('avg_F_1 =',avg_F_1)
   print('avg_F_2 =',avg_F_2)
   print('avg_F_3 =',avg_F_3)
   print('avg_F_4 =',avg_F_4)
   print('avg_F_5 =',avg_F_5)
   print('avg_F_6 =',avg_F_6)
   print('avg_F_7 =',avg_F_7)
   '''
   vesselness_np = vesselness.cpu().numpy()
   return vesselness_np / vesselness_np.max()