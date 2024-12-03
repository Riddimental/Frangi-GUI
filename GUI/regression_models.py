import numpy as np
import sounds
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

#Data
new_peaks_ellipsoid_14 = [0.6153846153846154, 0.6287625418060201, 0.6153846153846154, 0.6956521739130435, 0.8561872909698997, 0.7892976588628763, 1.0702341137123745, 0.9498327759197325, 1.0568561872909699, 1.0167224080267558, 1.7123745819397993]
new_peaks_ellipsoid_13 = [0.6270903010033445, 0.6521739130434783, 0.7274247491638797, 0.7274247491638797, 0.7525083612040134, 0.8277591973244147, 1.1036789297658862, 1.12876254180602, 1.028428093645485, 1.2792642140468227, 1.12876254180602]
new_peaks_ellipsoid_12 = [0.6688963210702341, 0.6688963210702341, 0.7224080267558528, 0.7357859531772575, 0.8561872909698997, 0.8160535117056856, 0.9096989966555183, 0.9899665551839465, 1.0836120401337792, 1.2709030100334449, 1.2575250836120402]
new_peaks_ellipsoid_11 = [0.8528428093645485, 0.8779264214046822, 0.8277591973244147, 1.153846153846154, 1.1789297658862876, 1.2290969899665551, 1.354515050167224, 1.254180602006689, 1.2792642140468227, 1.5050167224080269, 1.5050167224080269]
new_peaks_ellipsoid_10 = [0.9899665551839465, 0.9765886287625418, 0.9765886287625418, 1.1505016722408026, 1.1505016722408026, 1.177257525083612, 1.2709030100334449, 1.2709030100334449, 1.6187290969899666, 1.3377926421404682, 1.6989966555183946]
new_peaks_ellipsoid_9 = [1.511705685618729, 1.511705685618729, 1.5384615384615383, 1.4180602006688963, 1.511705685618729, 1.4180602006688963, 1.7926421404682273, 1.9130434782608696, 1.6989966555183946, 1.351170568561873, 1.7391304347826086]
new_peaks_ellipsoid_8 = [1.7307692307692308, 1.705685618729097, 1.7307692307692308, 1.705685618729097, 1.7558528428093645, 1.7307692307692308, 1.8812709030100334, 1.9063545150501673, 2.031772575250836, 1.8060200668896322, 1.9565217391304348]
new_peaks_ellipsoid_7 = [1.7926421404682273, 1.7792642140468227, 1.806020066889632, 1.7391304347826086, 1.7391304347826086, 1.9264214046822743, 1.8729096989966554, 1.9531772575250836, 1.765886287625418, 1.8595317725752507, 2.1672240802675584]
new_peaks_ellipsoid_6 = [2.2474916387959865, 2.2474916387959865, 2.140468227424749, 2.260869565217391, 2.2474916387959865, 2.140468227424749, 2.234113712374582, 2.3277591973244145, 2.2073578595317724, 2.568561872909699, 2.448160535117057]
new_peaks_ellipsoid_5 = [2.4832775919732444, 2.4832775919732444, 2.508361204013378, 2.4331103678929766, 2.5334448160535117, 2.5836120401337794, 2.5836120401337794, 2.508361204013378, 2.4832775919732444, 2.4832775919732444, 2.4832775919732444]
new_peaks_ellipsoid_4 = [2.608695652173913, 2.5819397993311037, 2.648829431438127, 2.5551839464882944, 2.782608695652174, 2.862876254180602, 2.688963210702341, 2.568561872909699, 2.528428093645485, 2.648829431438127, 2.4615384615384617]
new_peaks_ellipsoid_3 = [3.2357859531772575, 3.2357859531772575, 3.2357859531772575, 3.1354515050167224, 3.160535117056856, 3.160535117056856, 2.959866220735786, 3.0351170568561874, 3.1354515050167224, 3.1354515050167224, 3.18561872909699]
new_peaks_ellipsoid_2 = [3.9632107023411374, 3.9632107023411374, 3.9632107023411374, 3.8628762541806023, 3.6872909698996654, 3.6622073578595318, 3.7123745819397995, 3.737458193979933, 3.8377926421404682, 3.8377926421404682, 3.9130434782608696]
new_peaks_ellipsoid_1 = [4.590301003344481, 4.6404682274247495, 4.665551839464883, 4.439799331103679, 4.540133779264214, 4.615384615384616, 4.6404682274247495, 4.765886287625418, 4.590301003344481, 4.6404682274247495, 4.690635451505017]

# Definir tamaños de elipsoides (valores en sigma = 0)
sizes = np.array([new_peaks_ellipsoid_1[0],
                  new_peaks_ellipsoid_2[0],
                  new_peaks_ellipsoid_3[0],
                  new_peaks_ellipsoid_4[0],
                  new_peaks_ellipsoid_5[0],
                  new_peaks_ellipsoid_6[0],
                  new_peaks_ellipsoid_7[0],
                  new_peaks_ellipsoid_8[0],
                  new_peaks_ellipsoid_9[0],
                  new_peaks_ellipsoid_10[0],
                  new_peaks_ellipsoid_11[0],
                  new_peaks_ellipsoid_12[0],
                  new_peaks_ellipsoid_13[0],
                  new_peaks_ellipsoid_14[0]])

# Definir valores de sigma
sigmas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 
                   0.8, 0.9, 1.0])


# Crear una lista con todos los pares (tamaño, sigma) y sus correspondientes escalas
x = []
y = []

# Tus datos de escalas para cada elipsoide
scales = [new_peaks_ellipsoid_1, new_peaks_ellipsoid_2, new_peaks_ellipsoid_3, 
          new_peaks_ellipsoid_4, new_peaks_ellipsoid_5, new_peaks_ellipsoid_6, 
          new_peaks_ellipsoid_7, new_peaks_ellipsoid_8, new_peaks_ellipsoid_9, 
          new_peaks_ellipsoid_10, new_peaks_ellipsoid_11, new_peaks_ellipsoid_12, 
          new_peaks_ellipsoid_13, new_peaks_ellipsoid_14]

# Llenar X con pares (tamaño, snr) y y con los valores de escala correspondientes
for i, size in enumerate(sizes):
    for j, sigma in enumerate(sigmas):
        x.append([size, sigma])
        y.append(scales[i][j])

# Convertir a numpy arrays
x = np.array(x)
y = np.array(y)

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Transformar características para crear términos polinomiales
poly = PolynomialFeatures(degree=3)  # Cambia el grado según la complejidad deseada
X_poly = poly.fit_transform(X_scaled)

# Entrenar el modelo de regresión polinómica
model = LinearRegression()
model.fit(X_poly, y)




def predict_scale(snr:float, size:float) -> float:
    
    #this model was trained with ellipsoids of ROI intensity 1, so the noise sigma that relates the real image to the ellipsoids (snr) is 1/snr, snr = 1/noise
    
    #if snr is infity, the noise sigma is 0
    if snr == float('inf'):
        noise_sigma = 0
    else:
        #noise_sigma = 1/snr
        noise_sigma = 0.5/snr
    
    sounds.ai()
    # Escalar el nuevo dato
    X_new = scaler.transform([[size, noise_sigma]])

    # Transformar a términos polinomiales
    X_new_poly = poly.transform(X_new)

    # Realizar la predicción
    scale_pred = model.predict(X_new_poly)
    print(f"Optimal scale predicted for Voxels of {2*size:.2f} mm with snr {snr}: {scale_pred[0]:.3f}")
    
    return float(scale_pred[0])