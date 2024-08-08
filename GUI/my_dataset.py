import glob
import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage

# Define the 3D transformation classes
class Normalize3D:
   """Normalizes the input volume by subtracting the mean and dividing by the standard deviation."""
   def __init__(self, mean, std):
      self.mean = mean
      self.std = std

   def __call__(self, volume):
      return (volume - self.mean) / self.std

class RandomRotation3D:
   """Rotates the input volume randomly around the central axis."""
   def __init__(self, degrees):
      self.degrees = degrees

   def __call__(self, volume):
      angle = np.random.uniform(-self.degrees, self.degrees)
      return torch.tensor(scipy.ndimage.rotate(volume.numpy(), angle, axes=(1, 2), reshape=False))

class RandomCrop3D:
   """Randomly crops the input volume."""
   def __init__(self, crop_size):
      self.crop_size = crop_size

   def __call__(self, volume):
      z, x, y = volume.shape
      dz, dx, dy = self.crop_size
      start_z = np.random.randint(0, z - dz)
      start_x = np.random.randint(0, x - dx)
      start_y = np.random.randint(0, y - dy)
      return volume[start_z:start_z + dz, start_x:start_x + dx, start_y:start_y + dy]

class RandomFlip3D:
   """Randomly flips the input volume along the specified axes."""
   def __init__(self, axes):
      self.axes = axes

   def __call__(self, volume):
      for axis in self.axes:
         if np.random.rand() > 0.5:
               volume = torch.flip(volume, dims=[axis])
      return volume

class Transform3D:
   """Combines multiple 3D transformations into a single transformation."""
   def __init__(self):
      self.normalize = Normalize3D(mean=0.5, std=0.5)
      self.rotate = RandomRotation3D(degrees=30)
      #self.crop = RandomCrop3D(crop_size=(64, 64, 64))
      self.flip = RandomFlip3D(axes=[0, 1, 2])

   def __call__(self, volume):
      volume = self.normalize(volume)
      volume = self.rotate(volume)
      #volume = self.crop(volume)
      volume = self.flip(volume)
      return volume

# Implement a precise mapping between images and labels
def get_mapped_paths(base_dir):
   """
   Given a base directory, retrieves the paths of all NIfTI (.nii.gz) files in the 'IMAGES' and 'LABELS'
   subdirectories, and creates a mapping between the image paths and their corresponding label paths.
   The function assumes that the image file names follow a specific pattern with '_E_' and '_X_' separators,
   where 'E' represents the template ID and 'X' represents the case ID. The label file names are assumed to have
   a specific pattern with 'template_E_HR_PVS_mask_Case_X_Rep_1.nii.gz', where 'E' and 'X' are the same as in the
   image file names.

   Parameters:
   - base_dir (str): The base directory where the 'IMAGES' and 'LABELS' subdirectories are located.

   Returns:
   - mapping (List[Tuple[str, str]]): A list of tuples, where each tuple contains the path of an image file and its
   corresponding label file. If a label file is not found for a given image file, it is not included in the mapping.
   """
   image_dir = os.path.join(base_dir, 'IMAGES')
   label_dir = os.path.join(base_dir, 'LABELS')

   image_paths = glob.glob(os.path.join(image_dir, '**', '*.nii.gz'), recursive=True)
   label_paths = glob.glob(os.path.join(label_dir, '**', '*.nii.gz'), recursive=True)

   # Create a dictionary mapping images to labels
   mapping = []
   for image_path in image_paths:
      # Extract E and X from the image file name
      base_name = os.path.basename(image_path)
      parts = base_name.split('_')
      E, X = parts[1], parts[5]

      # Find the corresponding label using E and X
      label_pattern = f'template_{E}_HR_PVS_mask_Case_{X}_Rep_1.nii.gz'
      label_path = next((p for p in label_paths if label_pattern in p), None)

      if label_path:
         mapping.append((image_path, label_path))

   return mapping

# Create a dataset using the mapping
class MRIDataset(Dataset):
   """A dataset that loads NIfTI images and their corresponding labels."""
   def __init__(self, mapping, transform=None):
      self.mapping = mapping
      self.transform = transform

   def __len__(self):
      return len(self.mapping)

   def __getitem__(self, idx):
      image_path, label_path = self.mapping[idx]

      image = nib.load(image_path).get_fdata()
      label = nib.load(label_path).get_fdata()

      image = torch.tensor(image, dtype=torch.float32)
      label = torch.tensor(label, dtype=torch.float32)

      if self.transform:
         image, label = self.transform(image, label)
      return image, label

