from Environment import EnvironmentClass, Learn
from RLAlgorithm import ActorCriticAgent, DDPGAgent
from datetime import datetime
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time
import os

SEED = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main(LOGDIR = "logs", PROJECTNAME = "RL5000", RLALGORITHM = "DDGP", EPISODE = 100, VALIDATIONFREQ= 5, MODEL_DIR = "models", LearningRateA = 0.000005, LearningRateB = 0.0001):
    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"  

    runname = f'{RLALGORITHM}_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{EPISODE}_{LearningRateA}_{LearningRateB}'
    writer = SummaryWriter(log_dir=f'./{LOGDIR}/{PROJECTNAME}/{runname}')
    carenv = EnvironmentClass("Training", SEED=SEED)
    INPUTDIM = len(carenv.objectreturn)
    #agent = ActorCriticAgent(alpha=LearningRateA, beta = LearningRateB, input_dims=[INPUTDIM], gamma=0.9999, layer1_size=256, layer2_size=256, writer=writer, MODELSAVE=MODEL_DIR, FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}", seed=SEED)
    agent = DDPGAgent(alpha=LearningRateA, beta = LearningRateB, input_dims=[INPUTDIM], n_actions=2, writer=writer, FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}", seed=SEED, tau=0.001)
    Learn(agent, carenv, writer, VALIDATIONFREQ, EPISODE)

   


if __name__ == "__main__":
    start = time.time()
    
    main(EPISODE=500, VALIDATIONFREQ=10, LearningRateA = 0.00025, LearningRateB = 0.0025)
    main(EPISODE=500, VALIDATIONFREQ=10, LearningRateA = 0.0025, LearningRateB = 0.025)
    main(EPISODE=500, VALIDATIONFREQ=10, LearningRateA = 0.000025, LearningRateB = 0.00025)
    end = time.time()
    elapsed = end - start
    print(f'Time taken: {elapsed:.6f} seconds')