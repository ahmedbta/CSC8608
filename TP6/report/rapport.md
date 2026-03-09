# Compte-rendu TP6 — IA Explicable (XAI)

**Auteur :** Ahmed Ben Taleb Ali  
**Date :** 09 mars 2026

---

## 1. Analyse Grad-CAM (ResNet50)

J'ai testé le modèle sur des radiographies pour détecter des pneumonies. Voici ce que Grad-CAM affiche pour chaque image :

| Image | Prédiction | Grad-CAM | Temps (Inf / XAI) |
|---|---|---|---|
| normal_1.jpeg | **PNEUMONIA** (Erreur) | ![Normal 1](../outputs/gradcam_normal_1.png) | 0.06s / 0.04s |
| normal_2.jpeg | **NORMAL** | ![Normal 2](../outputs/gradcam_normal_2.png) | 0.05s / 0.04s |
| pneumo_1.jpeg | **NORMAL** (Erreur) | ![Pneumo 1](../outputs/gradcam_pneumo_1.png) | 0.05s / 0.04s |
| pneumo_2.jpeg | **PNEUMONIA** | ![Pneumo 2](../outputs/gradcam_pneumo_2.png) | 0.05s / 0.04s |

**Observations sur les erreurs :**
Sur l'image `normal_1`, le modèle se trompe. En regardant Grad-CAM, on voit qu'il n'analyse pas vraiment les poumons mais se focalise sur les bords de la radio et les contrastes sur les côtés. C'est l'effet **Clever Hans** : le modèle a appris des biais du dataset (comme des marques sur les films ou la position du patient) au lieu de chercher des signes médicaux.

**Qualité visuelle :**
Les zones sont floues et ressemblent à des gros blocs. C'est normal car Grad-CAM utilise la dernière couche du ResNet qui est très petite (environ 7x7). Quand on l'étire pour l'afficher sur l'image d'origine, ça fait cet effet de flou.

---

## 2. Integrated Gradients et SmoothGrad

Ici, on cherche une précision au pixel près.

| Image | Rendu IG & SmoothGrad |
|---|---|
| normal_1 | ![IG Normal 1](../outputs/ig_smooth_normal_1.png) |
| pneumo_2 | ![IG Pneumo 2](../outputs/ig_smooth_pneumo_2.png) |

**Vitesse et déploiement :**
- IG prend environ 2.6 secondes.
- SmoothGrad prend plus de **1 minute (75s)** car il doit calculer 25 versions de l'image.

Pour un médecin, c'est beaucoup trop long d'attendre 1 minute après chaque clic. Pour régler ça, il faudrait une architecture **asynchrone** : le médecin lance l'analyse, et une file d'attente (Task Queue) gère le calcul en arrière-plan pendant qu'il continue son travail, puis l'image s'affiche quand elle est prête.

**Mathématiques :**
L'avantage d'Integrated Gradients est qu'il peut montrer des valeurs négatives. Contrairement à Grad-CAM qui ne montre que ce qui "aide" la prédiction, IG peut montrer ce qui "contredit" la classe, ce qui est plus complet pour une analyse médicale.

---

## 3. Modèle simple (Régression Logistique)

J'ai utilisé une régression logistique sur des données tabulaires (Cancer du sein).

![Coefficients](../outputs/glassbox_coefficients.png)

- **Accuracy** : 97.37%
- **Variable la plus importante (Maligne)** : C'est **"worst texture"** qui a le plus gros impact négatif.
- **Pourquoi c'est mieux ?** C'est un modèle "Glass-box". L'explication est directe : on lit juste les poids (coefficients) du modèle. Il n'y a pas besoin de méthodes complexes pour comprendre comment il décide.

---

## 4. SHAP sur Random Forest

Enfin, j'ai testé SHAP sur un modèle plus complexe (Random Forest).

| Importance Globale (Summary) | Zoom sur le Patient 0 (Waterfall) |
|---|---|
| ![SHAP Summary](../outputs/shap_summary.png) | ![SHAP Waterfall](../outputs/shap_waterfall.png) |

**Analyse globale :**
Le Random Forest avec SHAP identifie à peu près les mêmes variables importantes que la régression logistique (texture, taille de la tumeur). Comme deux modèles différents trouvent les mêmes résultats, on peut dire que ces caractéristiques sont des **biomarqueurs fiables**.

**Analyse locale (Patient 0) :**
Pour ce patient, c'est la variable **"worst area"** (valeur : **677.9**) qui a le plus aidé à prédire que la tumeur était bénigne.
