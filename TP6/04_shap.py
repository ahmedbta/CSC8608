import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
import os

# 1. Chargement et Séparation des données
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # 0: Maligne, 1: Bénigne

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Entraînement d'une "Boîte Noire" (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
# TODO: Entraîner le modèle sur les données d'entraînement
model.fit(X_train, y_train)

print(f"Accuracy du Random Forest : {model.score(X_test, y_test):.4f}")

# 3. Explicabilité Post-Hoc avec SHAP
# TODO: Instancier le TreeExplainer de SHAP en lui passant le modèle entraîné
explainer = shap.TreeExplainer(model)

# TODO: Calculer les valeurs SHAP (Explanation object) pour l'ensemble du jeu de test.
shap_values = explainer(X_test)

# L'API de SHAP prédit par défaut la probabilité de la classe 1 (Bénigne) pour un RandomForest binaire.
# Nous allons extraire les valeurs SHAP spécifiques à cette classe (index 1) pour les visualisations.
shap_values_class1 = shap_values[:, :, 1]

# 4. Explicabilité Locale : Waterfall Plot (Un seul patient)
patient_idx = 0
plt.figure(figsize=(10, 6))
# show=False permet à matplotlib de sauvegarder l'image au lieu d'ouvrir une fenêtre graphique
shap.plots.waterfall(shap_values_class1[patient_idx], show=False)
plt.title(f"Explication Locale SHAP - Patient {patient_idx}")
plt.tight_layout()

# Sauvegarde dans le dossier outputs
script_dir = os.path.dirname(os.path.abspath(__file__))
output_local = os.path.join(script_dir, "outputs", "shap_waterfall.png")
plt.savefig(output_local, bbox_inches='tight')
plt.close()
print(f"Waterfall plot sauvegardé dans {output_local}")

# 5. Explicabilité Globale : Summary Plot
plt.figure(figsize=(10, 8))
# Le summary plot analyse toutes les prédictions du jeu de test en même temps
shap.summary_plot(shap_values_class1, X_test, show=False)
plt.title("Importance globale et directionnelle des variables (SHAP)")
plt.tight_layout()

output_global = os.path.join(script_dir, "outputs", "shap_summary.png")
plt.savefig(output_global, bbox_inches='tight')
plt.close()
print(f"Summary plot sauvegardé dans {output_global}")

# Small trick to get the specific feature and value for patient 0 for the report
sv = shap_values_class1[patient_idx]
# Find the feature with the highest contribution (absolute value of shap_value)
max_idx = sv.values.argmax() if sv.values.max() > abs(sv.values.min()) else sv.values.argmin()
feature_name = X_test.columns[max_idx]
feature_val = X_test.iloc[patient_idx, max_idx]
shap_val = sv.values[max_idx]
print(f"PATIENT_0_DEBUG: Feature={feature_name}, Value={feature_val:.4f}, SHAP={shap_val:.4f}")
