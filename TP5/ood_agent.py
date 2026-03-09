import gymnasium as gym
from stable_baselines3 import PPO
from PIL import Image
import os

print("--- ÉVALUATION OOD : GRAVITÉ FAIBLE ---")

# On récupère le dossier du script pour que les chemins soient relatifs à celui-ci
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "outputs", "ppo_lunar_lander")
output_dir = os.path.join(script_dir, "outputs")

# Création de l'environnement avec une gravité modifiée (simuler la Lune réelle)
# Par défaut la gravité est de -10.0. On passe à -2.0.
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=-2.0)

# Chargement du modèle entraîné à l'Exercice 2
if not os.path.exists(model_path + ".zip"):
    print(f"ERREUR : Le modèle est introuvable à l'adresse {model_path}.zip")
    exit(1)

model = PPO.load(model_path, device="cpu")

obs, info = eval_env.reset()
done = False
frames = []

total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    
    # Mise à jour des métriques
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1
        
    frames.append(Image.fromarray(eval_env.render()))
    done = terminated or truncated

eval_env.close()

# Analyse du vol
if reward == -100:
    issue = "CRASH DÉTECTÉ 💥"
elif reward == 100:
    issue = "ATTERRISSAGE RÉUSSI 🏆"
else:
    issue = "TEMPS ÉCOULÉ OU SORTIE DE ZONE ⚠️"

print("\n--- RAPPORT DE VOL PPO (GRAVITÉ MODIFIÉE) ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

if frames:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gif_path = os.path.join(output_dir, 'ood_agent.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=30, loop=0)
    print(f"Vidéo de la télémétrie sauvegardée sous '{gif_path}'")
