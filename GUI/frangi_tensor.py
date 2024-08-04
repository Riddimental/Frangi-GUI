import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.ndimage as ndi
import cv2
import torch.nn.functional as F

def apply_gaussian_blur_3d(image, sigma):
   """
   Apply 3D Gaussian blur to the image using scipy's gaussian_filter.
   """
   # Ensure image is on CPU for scipy operations
   image = image.cpu().numpy()
   
   # Apply Gaussian filter
   blurred_image = ndi.gaussian_filter(image, sigma=sigma)
   
   # Convert back to torch tensor
   blurred_image = torch.tensor(blurred_image, dtype=torch.float32, device=image.device)
   return blurred_image

def divide_nonzero(a, b):
   """
   Divide elements of a by b, handling division by zero.
   """
   result = torch.where(b != 0, a / b, torch.zeros_like(a))
   return result

def compute_vesselness(eigvals, alpha, beta):
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

def compute_eig_vals(hessian):
   # Convert the tensor to a NumPy array
   hessian_np = hessian.cpu().numpy()
   
   # Compute eigenvalues
   eigvals = np.linalg.eigvals(hessian_np.transpose(2, 3, 4, 0, 1))  # Move the 3x3 matrices to the last dimensions

   # Sort eigenvalues in descending order
   sorted_eigvals = -np.sort(-eigvals, axis=-1)
   print("evigvals done")
   return torch.from_numpy(sorted_eigvals)

def compute_hessian_return_eigvals(image, sigma=1):
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

def my_frangi_filter(input_image, scale_range=(1, 10), alpha=1, beta=0.5, steps=2, black_vessels=True):
   tensor_image = torch.tensor(input_image, dtype=torch.float64)  # Asegura que la imagen sea de tipo float64
   tensor_image /= tensor_image.max()  # Normaliza la imagen

   if black_vessels:
      tensor_image = -tensor_image
      print('image inverted')
      #image = intensity_rescale(image, 0, up_limit)  # Invertir y reescalar la imagen

   division = (scale_range[1] - scale_range[0]) / steps
   vesselness = torch.zeros_like(tensor_image)
   print('Scales from', scale_range[0], 'to', scale_range[1], 'in', steps, 'steps.')

   scale = scale_range[0]
   while scale <= scale_range[1]:
      print('Current scale:', scale)
      eigenvalues = compute_hessian_return_eigvals(tensor_image, sigma=scale)
      output = compute_vesselness(eigenvalues, alpha, beta).real
      vesselness += output / output.max()
      scale += division

   print("Frangi filter applied.")
   vesselness_np = vesselness.cpu().numpy()
   return vesselness_np / vesselness_np.max()
