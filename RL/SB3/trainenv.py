from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from carenvSB3 import CarEnv
import numpy as np
import logging

logging.getLogger().setLevel(logging.WARNING)

# Instantiate the environment
env = CarEnv("train")

# Save the best model checkpoint
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='td3_car_model')

# Evaluate the model every few episodes
eval_env = CarEnv("eval")  # Separate environment for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path='./models/best_model/',
                             log_path='./logs/', eval_freq=5000, deterministic=True, render=True)

# Define action noise for exploration
n_actions = env.action_space.shape[0]  # Number of continuous actions
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initialize TD3 agent
model = TD3(
    "MlpPolicy",               # Policy type
    env,                       # Environment
    verbose=1,                 # Verbosity
    learning_rate=1e-3,        # Learning rate
    buffer_size=int(1e6),      # Replay buffer size
    batch_size=512,            # Mini-batch size for learning
    gamma=0.99,                # Discount factor
    tau=0.005,                 # Target smoothing coefficient
    train_freq=1,              # Train after every step
    gradient_steps=1,          # Number of gradient steps per update
    action_noise=action_noise, # Add noise for exploration
    policy_delay=2,            # Delayed policy updates for stability
    target_policy_noise=0.2,   # Noise added to target policy during updates
    target_noise_clip=0.5,     # Limit the target noise
    device="cuda",             # Use GPU for training (if available)
    tensorboard_log="./tensorboard_logs/"  # Directory for TensorBoard logs
)


env.reset()
# Train the model
timesteps = 100000  # Total training timesteps
model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save("td3_car_model_final")

# # Evaluate the model
# obs = env.reset()
# for i in range(1000):
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     if done:
#         obs = env.reset()

print("Training complete and model evaluation finished.")
