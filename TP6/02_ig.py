import time
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from captum.attr import IntegratedGradients, NoiseTunnel
import os

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    def forward(self, x):
        return self.model(x).logits

# 1. Chargement de l'image et du modèle
image_path = sys.argv[1] if len(sys.argv) > 1 else "normal_1.jpeg"
if not os.path.exists(image_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    potential_path = os.path.join(script_dir, image_path)
    if os.path.exists(potential_path):
        image_path = potential_path

print(f"Analyse fine au pixel sur : {image_path}")
image = Image.open(image_path).convert("RGB")

model_name = "Aunsiels/resnet-pneumonia-detection"
processor = AutoImageProcessor.from_pretrained(model_name)
hf_model = AutoModelForImageClassification.from_pretrained(model_name)
wrapped_model = ModelWrapper(hf_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wrapped_model.to(device)
wrapped_model.eval()

inputs = processor(images=image, return_tensors="pt")
input_tensor = inputs["pixel_values"].to(device)
input_tensor.requires_grad = True

# Warm-up (Cold Start fix)
_ = wrapped_model(input_tensor)

# Inférence propre
start_infer = time.time()
logits = wrapped_model(input_tensor)
predicted_class_idx = logits.argmax(-1).item()
end_infer = time.time()

# 2. Integrated Gradients
ig = IntegratedGradients(wrapped_model)

# IG nécessite une image de référence neutre (baseline).
# Créez un tenseur rempli de zéros ayant exactement la même forme que l'input_tensor
baseline = torch.zeros_like(input_tensor)

start_ig = time.time()
# Lancer l'attribution IG avec 50 étapes (n_steps) et limiter la mémoire (internal_batch_size=2)
attributions_ig = ig.attribute(
    input_tensor, 
    baselines=baseline, 
    target=predicted_class_idx, 
    n_steps=50, 
    internal_batch_size=2
)
end_ig = time.time()

# 3. SmoothGrad (Via NoiseTunnel)
# Envelopper votre instance "ig" dans un NoiseTunnel
noise_tunnel = NoiseTunnel(ig)

start_sg = time.time()
# Génération de samples bruités (nt_samples) autour de l'image.
# On utilise 25 samples pour que l'exécution reste raisonnable en temps de démo, 
# tout en montrant l'effet de lissage.
attributions_sg = noise_tunnel.attribute(
    input_tensor, 
    nt_samples=25, 
    nt_type='smoothgrad', 
    target=predicted_class_idx,
    stdevs=0.1,
    internal_batch_size=2
)
end_sg = time.time()

print(f"Temps IG pur : {end_ig - start_ig:.4f}s")
print(f"Temps SmoothGrad (IG x 25) : {end_sg - start_sg:.4f}s")

# 4. Visualisation (Valeur absolue et filtrage du bruit)
# On prend la valeur absolue pour mesurer l'importance globale du pixel (qu'elle soit positive ou négative)
attr_ig_vis = np.sum(np.abs(attributions_ig.squeeze().cpu().detach().numpy()), axis=0)
attr_sg_vis = np.sum(np.abs(attributions_sg.squeeze().cpu().detach().numpy()), axis=0)

# Seuillage stochastique : on met à zéro tous les pixels dont l'importance est inférieure au 70e centile
threshold_ig = np.percentile(attr_ig_vis, 90) # On augmente le seuil pour IG car c'est très bruité
attr_ig_vis[attr_ig_vis < threshold_ig] = 0

threshold_sg = np.percentile(attr_sg_vis, 90)
attr_sg_vis[attr_sg_vis < threshold_sg] = 0

# Normalisation pour l'affichage
vmax_ig = np.max(attr_ig_vis)
vmax_sg = np.max(attr_sg_vis)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Affichage de l'image de base sur le premier axe
original_resized = image.resize(input_tensor.shape[2:][::-1])
axes[0].imshow(original_resized)
axes[0].set_title("Image Originale")
axes[0].axis('off')

# Superposition IG sur l'image originale
axes[1].imshow(original_resized, alpha=0.6)
# Utilisation de la colormap 'hot' (noir vers rouge/jaune/blanc)
axes[1].imshow(attr_ig_vis, cmap='hot', alpha=0.6, vmin=0, vmax=vmax_ig)
axes[1].set_title("Integrated Gradients (Seuillé)")
axes[1].axis('off')

# Superposition SmoothGrad sur l'image originale
axes[2].imshow(original_resized, alpha=0.6)
axes[2].imshow(attr_sg_vis, cmap='hot', alpha=0.6, vmin=0, vmax=vmax_sg)
axes[2].set_title("SmoothGrad (Seuillé)")
axes[2].axis('off')

# Sauvegarde
script_dir = os.path.dirname(os.path.abspath(__file__))
base_name = os.path.basename(image_path).split('.')[0]
output_filename = os.path.join(script_dir, "outputs", f"ig_smooth_{base_name}.png")

plt.savefig(output_filename, bbox_inches='tight')
plt.close(fig)
print(f"Visualisation sauvegardée dans {output_filename}")
