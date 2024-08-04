import torch
import numpy as np
import torch.optim as optim

def loss_iou(pred, target, threshold=0.5): #0.5 temporary
   pred = (pred > threshold).float()
   target = (target > threshold).float()
   intersection = (pred * target).sum()
   union = pred.sum() + target.sum() - intersection
   iou = intersection / union
   return 1 - iou  # Minimizes the distance between the result and the ground truth

# Supongamos que 'params' son los parámetros del filtro de Frangi
optimizer = optim.Adam(params, lr=0.01)

def train(model, dataloader, optimizer, num_epochs=10):
   model.train()
   for epoch in range(num_epochs):
      running_loss = 0.0
      for images, labels in dataloader:
         # Aplicar el filtro de Frangi
         outputs = model(images)
         
         # Calcular la pérdida
         loss = loss_iou(outputs, labels)
         
         # Backward y optimización
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         
         running_loss += loss.item() * images.size(0)
      epoch_loss = running_loss / len(dataloader.dataset)
      print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
