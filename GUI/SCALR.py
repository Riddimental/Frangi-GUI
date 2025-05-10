# SCALR.py
import pandas as pd
import numpy as np
import nibabel as nib
import filters as ft
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Data
data = pd.read_csv('results.csv')

# Display random sample of data
#print("Random samples from the dataset:")
#print(data.sample(10))

# Replace infinite values in SNR with a large number and apply log transformation
data['SNR'] = data['SNR'].replace([np.inf, -np.inf], 1000)
data['SNR'] = np.log1p(data['SNR'])  # Apply log transformation

# Prepare features and target
X_alpha = data[['SNR', 'Voxel Size (mm)']]
y_alpha = data['Best Scale (mm)']

# Apply polynomial features transformation
degree = 3 # You can experiment with different degrees
poly = PolynomialFeatures(degree)
X_poly_alpha = poly.fit_transform(X_alpha)

# Train-Test Split
X_train_alpha, X_test_alpha, y_train_alpha, y_test_alpha = train_test_split(X_poly_alpha, y_alpha, test_size=0.2, random_state=42)

# Train Model
model_alpha = LinearRegression()
model_alpha.fit(X_train_alpha, y_train_alpha)

# Evaluate Model
pred_alpha = model_alpha.predict(X_test_alpha)
mse_alpha = mean_squared_error(y_test_alpha, pred_alpha)
r2_alpha = r2_score(y_test_alpha, pred_alpha)

print("SCALR-alpha Polynomial Model - Mean Squared Error:", mse_alpha)
print("SCALR-alpha Polynomial Model - R² Score:", r2_alpha)

# Assuming pred_alpha is a list or array of predicted values
comparison_df = pd.DataFrame({
    'SNR': X_test_alpha[:, 3],  # Original SNR values
    'Voxel Size (mm)': X_test_alpha[:, 2],  # Original Voxel Size (mm) values
    'Actual Best Scale (mm)': y_test_alpha.values,  # Use values attribute for Series
    'Predicted Best Scale (mm)': pred_alpha
})

# Display the DataFrame to check correctness
print(comparison_df.head())

# Predict Best Scale for New Data in mm
def predict(image: nib.Nifti1Image) -> float:
   #obtain the SNR from the NIFTI image
   noise, snr = ft.get_sigma_and_snr(image.get_fdata())
   #make sure the NIFTI's voxels are isometric
   image = ft.isometric_voxels(image)
   #obtain the voxel size
   voxel_size = image.header.get_zooms()[0]
   # Replace large or infinite SNR values
   if np.isinf(snr) or snr > 1000:
      snr = 1000
   snr = np.log1p(snr)  # Apply log transformation

   # Transform input data for polynomial features
   X_new = poly.transform([[snr, voxel_size]])
   return model_alpha.predict(X_new)[0]
