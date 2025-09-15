from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from carenvSB3 import CarEnv
import numpy as np
import logging


"""
Későbbiekben:
- Mekkora a biciklis sebessége: float


"""



#logging.getLogger().setLevel(logging.WARNING)

# Instantiate the environment
env = CarEnv("train")

# Save the best model checkpoint
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='DDPG_car_model')

# Evaluate the model every few episodes
eval_env = env  # Separate environment for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model/',
                             log_path='./logs/', eval_freq=500, deterministic=True, render=False)

# Define action noise for exploration
n_actions = env.action_space.shape[0]  # Number of continuous actions
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize DDPG agen
model = DDPG(
    "MlpPolicy",               # Policy type
    env,                       # Environment
    verbose=1,
    device="cpu",             # Use GPU for training (if available)
    tensorboard_log="./tensorboard_logs/"  # Directory for TensorBoard logs
)

# Train the model
timesteps = 150_000  # Total training timesteps
model.learn(total_timesteps=timesteps, progress_bar=True, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save("DDPG_car_model_final")

# # Evaluate the model
# obs = env.reset()
# for i in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()

print("Training complete and model evaluation finished.")
