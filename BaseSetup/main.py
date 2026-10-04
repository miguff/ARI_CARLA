import os
import time
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from Environment import EnvironmentClass, Learn
from RLAlgorithm import ActorCriticAgent, DDPGAgent, PPOAgent

SEED = 42

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(LOGDIR="logs", PROJECTNAME="TDK_TESZT", RLALGORITHM="PPO",
         EPISODE=100, VALIDATIONFREQ=5, MODEL_DIR="models",
         LearningRateA=3e-4, LearningRateB=1e-3,
         use_gt_distance=False):
    """Train an RL agent for longitudinal control in CARLA.

    Parameters
    ----------
    use_gt_distance : bool
        If True, use CARLA ground-truth actor distance instead of stereo
        vision.  This is the upper-bound ablation baseline.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    runname = (
        f'{RLALGORITHM}_{datetime.now().strftime("%Y%m%d-%H%M%S")}_'
        f'{EPISODE}_{LearningRateA}_{LearningRateB}'
        f'{"_GT" if use_gt_distance else "_stereo"}'
    )
    writer = SummaryWriter(log_dir=f'./{LOGDIR}/{PROJECTNAME}/{runname}')

    carenv = EnvironmentClass(
        "Training", SEED=SEED, model_type=RLALGORITHM,
        use_gt_distance=use_gt_distance,
    )
    INPUTDIM = len(carenv.objectreturn)

    if RLALGORITHM == "DDPG":
        agent = DDPGAgent(
            alpha=LearningRateA, beta=LearningRateB, input_dims=[INPUTDIM],
            n_actions=1, writer=writer,
            FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}",
            seed=SEED, tau=0.001
        )
    elif RLALGORITHM == "PPO":
        agent = PPOAgent(
            alpha=LearningRateA, beta=LearningRateB, input_dims=[INPUTDIM],
            n_actions=1, writer=writer, model_dir=MODEL_DIR,
            FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}",
            seed=SEED
        )
    elif RLALGORITHM == "ActorCritic":
        agent = ActorCriticAgent(
            alpha=LearningRateA, beta=LearningRateB, input_dims=[INPUTDIM],
            gamma=0.99, layer1_size=256, layer2_size=256, writer=writer,
            MODELSAVE=MODEL_DIR,
            FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}",
            seed=SEED
        )
    else:
        raise ValueError(f"Unknown RL algorithm: {RLALGORITHM}")

    Learn(agent, carenv, writer, VALIDATIONFREQ, EPISODE)


if __name__ == "__main__":
    start = time.time()

    # Main experiment: PPO with stereo vision
    main(EPISODE=260, VALIDATIONFREQ=10, RLALGORITHM="PPO",
         PROJECTNAME="TDK_FINAL", LearningRateA=0.0003, LearningRateB=0.001)

    # Uncomment for ground-truth ablation:
    # main(EPISODE=260, VALIDATIONFREQ=10, RLALGORITHM="PPO",
    #      PROJECTNAME="TDK_GT_ABLATION", LearningRateA=0.0003, LearningRateB=0.001,
    #      use_gt_distance=True)

    end = time.time()
    print(f'\nTime taken: {end - start:.2f} seconds')
