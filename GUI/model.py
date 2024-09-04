import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import my_dataset
import frangi_tensor
import filters
import numpy as np
from skimage.transform import resize

class DiceLoss(nn.Module):
   """
   Dice Loss function.
   """
   def __init__(self):
      super(DiceLoss, self).__init__()

   def forward(self, pred, target):
      """
      Calculates the Dice Loss between the predicted and target tensors.

      Parameters:
      - pred: predicted tensor
      - target: target tensor

      Returns:
      - Dice Loss: float value between 0 and 1
      """
      smooth = 1.0
      intersection = (pred * target).sum()
      dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
      return 1 - dice  # Dice Loss = 1 - Dice Coefficient

class FrangiParameterNet(nn.Module):
   """
   Neural network model for Frangi filter parameters estimation.
   """
   def __init__(self):
      super(FrangiParameterNet, self).__init__()
      self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
      self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
      self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
      self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
      self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Global pooling for variable size
      self.fc1 = nn.Linear(128, 512)  # Number of channels after convolutions
      self.fc2 = nn.Linear(512, 4)  # 4 parameters: scale, beta1, beta2, blackVessels
      #self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout probability


   def forward(self, x):
      """
      Forward pass of the model.

      Parameters:
      - x: input tensor

      Returns:
      - output tensor
      """
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = self.global_pool(x)  # Global pooling
      x = x.view(x.size(0), -1)  # Flatten the tensor
      x = F.relu(self.fc1(x))
      #x = self.dropout(x)  # Apply dropout
      x = self.fc2(x)
      return x

# Initialize the model
model = FrangiParameterNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
criterion = DiceLoss()  # Dice Loss

# Get the paths of the dataset
base_dir = '/Volumes/RIDDIMENTAL/MRI/DATASET'
image_paths, label_paths = my_dataset.get_mapped_paths(base_dir)

# Initialize the dataset and DataLoader
dataset = my_dataset.MRIDataset(image_paths, label_paths, transform=my_dataset.Transform3D())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def apply_frangi_filter(nib_file, params): #nib_file is a .nii file loaded from the dataset
   """
   Apply the Frangi filter to the input image based on the user's input parameters.

   Parameters:
   - nib_file: a .nii file loaded from the dataset
   - params: a tensor of shape (4,) containing the parameters for the Frangi filter

   Returns:
   - filtered_image: a numpy array containing the result of applying the Frangi filter
   """
   # Load the image data from the nib file
   image_data = nib_file.get_fdata().astype(np.float64)
   image_data /= np.max(image_data) # Normalize the image

   # Access the header metadata
   header = nib_file.header
   # Get voxel sizes
   voxel_sizes = header.get_zooms()
   min_voxel_size = min(voxel_sizes)
   voxel_size = min_voxel_size
   # Resample the image data to ensure cubic voxels
   if not all(v == voxel_sizes[0] for v in voxel_sizes): # Need to resize
         resampled_image = filters.resample_image(image_data, min_voxel_size)
         image_data = resampled_image
         print("Image resampled")

   # Apply a transform to ensure the parameters are within the correct range
   scale = torch.sigmoid(params[0]) * 4  # Scale in the range [0, 4] (in mm, a later conversion changes mm to voxels depending on the image proportions)
   beta1 = torch.sigmoid(params[1])      # Beta1 in the range [0, 1]
   beta2 = torch.sigmoid(params[2])      # Beta2 in the range [0, 1]
   blackVessels = torch.sigmoid(params[3]) > 0.5  # Boolean based on the threshold value

   # Apply the Frangi filter with these parameters
   filtered_image = frangi_tensor.my_frangi_filter(image_data, filters.mm2voxel(scale.item(), voxel_size), beta1.item(), beta2.item(), blackVessels.item())
   
   return filtered_image

for epoch in range(num_epochs):
   model.train()  # Training mode
   running_loss = 0.0

   for i, (images, labels) in enumerate(dataloader):
      optimizer.zero_grad()

      # Pass the images through the model to predict the parameters
      predicted_params = model(images)

      # Apply the Frangi filter with the predicted parameters
      filtered_image = apply_frangi_filter(images, predicted_params)

      # Calculate the loss
      loss = criterion(filtered_image, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

   print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

