from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from carenvTurning import CarEnv
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
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/Turning/', name_prefix='PPO_Turning3_model')

# Evaluate the model every few episodes
eval_env = CarEnv("eval")  # Separate environment for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path='./models/Turning/best_model/',
                             log_path='./logs/Turning/', eval_freq=5000, deterministic=True, render=False)

# Define action noise for exploration
n_actions = env.action_space.shape[0]  # Number of continuous actions
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

#vec_env = make_vec_env(env, 2)
# Initialize DDPG agen
model = PPO(
    "MlpPolicy",               # Policy type
    env,                       # Environment
    verbose=1,
    device="cpu",             # Use GPU for training (if available)
    tensorboard_log="./Turning/tensorboard_logs/"  # Directory for TensorBoard logs
)

# Train the model
timesteps = 100000  # Total training timesteps
model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save("PPO_car_Turning_final")

# # Evaluate the model
obs = env.reset()
km, distance, throttel, brake, turn_raius = obs[0]
obs = [km, distance, throttel, brake, turn_raius]

for i in range(100):
    
    action, _ = model.predict(obs, deterministic=True)
    print(f"Action: {action}")
    obs, reward, done, terminated, info  = env.step(action)
    print(f"Action reward: {reward}")
    if done:
        obs = env.reset()

print("Training complete and model evaluation finished.")
