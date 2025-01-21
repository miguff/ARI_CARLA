from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from carenvSB3 import CarEnv
import numpy as np
import logging


"""
Szóval amit csinálni kell, hogy jobban működjön:

Kell csinálni egy NN modeltt, hogy az adja ki, hogy fékezzen az autó vagy ne fékezzen.
Ezt meg kell tanítani és utána implementálni.
Bemenő paraméterek:
- Akar kanyarodni az autó: 1 v 0
- Milyen mesze van a biciklis: float
- Mekkora az auto sebessége: float

Meg kell tanítani a modell erre, itt szintetikusan létre kell hozni neki egy adatbázist amit tud tanulni, train-test-val-ra.
Ennek a modellnek 2 kimenete lesz.
- Fékezzen vagy nefékezzen: 1 v 0

Ha 1. akkor a kocsinak fékeznie kell, ebben az esetben az RL-nek csak 1 paramétert kell adni, hogy mekkora erővel fékezzen 0 és egy között
Ha 0 akkor a kocsinak folytatni kell az úthák, ekkor azt adja vissza a modell, hogy fékezzen-e vagy gyorsuljon. azért mert ha túl lépi a sebességet
akkor is kell neki fékezni. - Ezt még lehet bele lehet építeni a modellbe, de most még úgy látom, hogy azt egyszerűbb és biztosabb ha az RL találja meg.



Későbbiekben:
- Mekkora a biciklis sebessége: float


"""



logging.getLogger().setLevel(logging.WARNING)

# Instantiate the environment
env = CarEnv("train")

# Save the best model checkpoint
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='DDPG_car_model')

# Evaluate the model every few episodes
eval_env = CarEnv("eval")  # Separate environment for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model/',
                             log_path='./logs/', eval_freq=5000, deterministic=True, render=False)

# Define action noise for exploration
n_actions = env.action_space.shape[0]  # Number of continuous actions
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#vec_env = make_vec_env(env, 2)
# Initialize DDPG agen
model = DDPG(
    "MlpPolicy",               # Policy type
    env,                       # Environment
    verbose=1,
    device="cpu",             # Use GPU for training (if available)
    tensorboard_log="./tensorboard_logs/"  # Directory for TensorBoard logs
)

# Train the model
timesteps = 100000  # Total training timesteps
model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save("PPO_car_model_final")

# # Evaluate the model
# obs = env.reset()
# for i in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()

print("Training complete and model evaluation finished.")
