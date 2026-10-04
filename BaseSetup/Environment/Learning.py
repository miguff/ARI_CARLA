from RLAlgorithm import ActorCriticAgent, DDPGAgent, PPOAgent
from Environment import EnvironmentClass
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def Learn(agent, carenv, writer, VALIDATIONFREQ=5, EPISODE=100):
    """Training loop with contiguous MDP — RL controls every step.

    No more PID/RL handoff.  Every step is an RL step, so the buffer
    contains a proper contiguous trajectory for GAE.
    """
    REWARDS = []
    TRAINING_STEP = 0
    VALIDATION_STEP = 0

    for episode in range(EPISODE):
        is_validation = (episode % VALIDATIONFREQ == 0 and episode != 0)
        mode = "Validation" if is_validation else "Training"
        print(f"\n=== {mode} Episode {episode} ===")

        observation, reward, done, terminated = carenv.reset()
        EPISODE_TOTAL_REWARD = 0.0
        step_count = 0

        while not done:
            # Check for NaN in observation
            if any(np.isnan(v.item()) for v in observation):
                print(f"NaN in observation at step {step_count}, aborting episode")
                break

            # RL always chooses the action (no more PID handoff)
            action = agent.choose_action(observation, store=not is_validation)
            observation_, reward, done, terminated = carenv.step(action, training=not is_validation)

            EPISODE_TOTAL_REWARD += reward
            step_count += 1

            if is_validation:
                writer.add_scalar("Validation Step Reward", reward, VALIDATION_STEP)
                agent.learn(observation, action, reward, observation_, done,
                            TRAINING_STEP, Train=0)
                VALIDATION_STEP += 1
            else:
                agent.learn(observation, action, reward, observation_, done,
                            TRAINING_STEP, Train=1)
                writer.add_scalar("Training Step Reward", reward, TRAINING_STEP)
                TRAINING_STEP += 1

            writer.flush()
            observation = observation_

        # Episode end
        if is_validation:
            writer.add_scalar("Validation Episode Reward", EPISODE_TOTAL_REWARD, episode)
            writer.add_scalar("Validation Episode Steps", step_count, episode)
            agent.save_models(str(round(EPISODE_TOTAL_REWARD, 2)))
        else:
            writer.add_scalar("Training Episode Reward", EPISODE_TOTAL_REWARD, episode)
            writer.add_scalar("Training Episode Steps", step_count, episode)

        print(f"  Reward: {EPISODE_TOTAL_REWARD:.2f}  Steps: {step_count}")
        writer.flush()
        REWARDS.append(EPISODE_TOTAL_REWARD)

    carenv.cleanup()
    writer.close()

    # Print summary
    print(f"\n=== Training Summary ===")
    print(f"Total episodes: {EPISODE}")
    print(f"Mean reward (last 20): {np.mean(REWARDS[-20:]):.2f}")
    print(f"Mean reward (all):     {np.mean(REWARDS):.2f}")
