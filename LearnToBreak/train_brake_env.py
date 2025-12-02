from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from BrakingLearning import CarEnv
import numpy as np
import logging



logging.getLogger().setLevel(logging.WARNING)

# Instantiate the environment
env = CarEnv("train")

MODELNAME = "PPO"

# Save the best model checkpoint
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f'./models/{MODELNAME}_braking', name_prefix=f'{MODELNAME}_model')
# Evaluate the model every few episodes
eval_env = CarEnv("eval")  # Separate environment for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=f'./models/{MODELNAME}_braking/best_model/',
                             log_path=f'./logs/{MODELNAME}_braking/', eval_freq=5000, deterministic=True, render=False)

# Define action noise for exploration
n_actions = env.action_space.shape[0]  # Number of continuous actions
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = PPO(
    "MlpPolicy",               # Policy type
    env,                       # Environment
    verbose=1,
    device="cpu",             # Use GPU for training (if available)
    tensorboard_log="./tensorboard_logs/"  # Directory for TensorBoard logs
)

timesteps = 100_000  # Total training timesteps
model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])
model.save(f"{MODELNAME}_breaking_final")

print("Training complete and model evaluation finished.")