import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import my_dataset
import frangi_tensor

class DiceLoss(nn.Module):
   def __init__(self):
      super(DiceLoss, self).__init__()

   def forward(self, pred, target):
      smooth = 1.0
      intersection = (pred * target).sum()
      dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
      return 1 - dice  # Dice Loss = 1 - Dice Coefficient

class FrangiParameterNet(nn.Module):
   def __init__(self):
      super(FrangiParameterNet, self).__init__()
      self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
      self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1)
      self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
      self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
      self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Pooling global para tamaño variable
      self.fc1 = nn.Linear(128, 512)  # Número de canales después de las convoluciones
      self.fc2 = nn.Linear(512, 4)  # 4 parámetros: scale, beta1, beta2, blackVessels
      #self.dropout = nn.Dropout(p=0.5)  # Dropout layer with 50% dropout probability


   def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = self.pool(F.relu(self.conv3(x)))
      x = self.global_pool(x)  # Pooling global
      x = x.view(x.size(0), -1)  # Aplana el tensor
      x = F.relu(self.fc1(x))
      #x = self.dropout(x)  # Apply dropout
      x = self.fc2(x)
      return x

# Inicializar el modelo
model = FrangiParameterNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
criterion = DiceLoss()  # Usamos Dice Loss

# Obtener las rutas del dataset
base_dir = '/Volumes/RIDDIMENTAL/MRI/DATASET'
image_paths, label_paths = my_dataset.get_mapped_paths(base_dir)

# Inicializar el Dataset y DataLoader
dataset = my_dataset.MRIDataset(image_paths, label_paths, transform=my_dataset.Transform3D())
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

def apply_frangi_filter(image, params):
   # Aplicar una transformación para asegurarse de que los parámetros están en el rango correcto
   scale = torch.sigmoid(params[0]) * 4  # Scale en el rango [0, 4]
   beta1 = torch.sigmoid(params[1])      # Beta1 en el rango [0, 1]
   beta2 = torch.sigmoid(params[2])      # Beta2 en el rango [0, 1]
   blackVessels = torch.sigmoid(params[3]) > 0.5  # Booleano basado en el valor umbral

   # Aplicar el filtro Frangi con estos parámetros
   filtered_image = frangi_tensor.my_frangi_filter(image, scale.item(), beta1.item(), beta2.item(), blackVessels.item())
   
   return filtered_image

for epoch in range(num_epochs):
   model.train()  # Modo entrenamiento
   running_loss = 0.0

   for i, (images, labels) in enumerate(dataloader):
      optimizer.zero_grad()

      # Pasar las imágenes por el modelo para predecir los parámetros
      predicted_params = model(images)

      # Aplicar el filtro Frangi con los parámetros predichos
      filtered_image = apply_frangi_filter(images, predicted_params)

      # Calcular la pérdida
      loss = criterion(filtered_image, labels)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

   print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")
