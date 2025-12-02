from Environment import EnvironmentClass, Learn
from RLAlgorithm import ActorCriticAgent, DDPGAgent, PPOAgent
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


"""
Maybe I can go a this, that I manually simulate some environment. Like I move the car with keys (W,A,S,D) and store every input variable, my speed, distance to other cars, use sensors (Lane detection, RSS (saftey), Lidaer) and Cameras.
And After I have some run, I can run a learing algorithm on this. 
Need to have some second thought on that. 

"""




def main(LOGDIR = "logs", PROJECTNAME = "TDK_TESZT", RLALGORITHM = "DDPG", EPISODE = 100, VALIDATIONFREQ= 5, MODEL_DIR = "models", LearningRateA = 3e-4, LearningRateB = 1e-3):
    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\Graphviz\\bin"  

    runname = f'{RLALGORITHM}_{datetime.now().strftime("%Y%m%d-%H%M%S")}_{EPISODE}_{LearningRateA}_{LearningRateB}'
    writer = SummaryWriter(log_dir=f'./{LOGDIR}/{PROJECTNAME}/{runname}')
    carenv = EnvironmentClass("Training", SEED=SEED, model_type=RLALGORITHM)
    INPUTDIM = len(carenv.objectreturn)
    
    if RLALGORITHM == "DDPG":
        agent = DDPGAgent(alpha=LearningRateA, beta = LearningRateB, input_dims=[INPUTDIM], n_actions=1, writer=writer, FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}", seed=SEED, tau=0.001)
    elif RLALGORITHM == "PPO":
        agent = PPOAgent(alpha=LearningRateA, beta = LearningRateB, input_dims=[INPUTDIM],writer=writer, FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}", seed=SEED)
    elif RLALGORITHM == "ActorCritic":
        agent = ActorCriticAgent(alpha=LearningRateA, beta = LearningRateB, input_dims=[INPUTDIM], gamma=0.9999, layer1_size=256, layer2_size=256, writer=writer, MODELSAVE=MODEL_DIR, FilenamePrefix=f"{RLALGORITHM}_{EPISODE}_{LearningRateA}_{LearningRateB}", seed=SEED)

    agent.export_to_onnx()

    #Learn(agent, carenv, writer, VALIDATIONFREQ, EPISODE)

   


if __name__ == "__main__":
    start = time.time()
    
    # #main(EPISODE=400, VALIDATIONFREQ=10, RLALGORITHM="DDPG", PROJECTNAME="TESZTING_DDPG")
    # #main(EPISODE=400, VALIDATIONFREQ=10, RLALGORITHM="DDPG", PROJECTNAME="TESZTING_DDPG", LearningRateA = 0.00003, LearningRateB = 0.0001)
    # #main(EPISODE=400, VALIDATIONFREQ=10, RLALGORITHM="PPO", PROJECTNAME="TESZTING_PPO", LearningRateA = 0.003, LearningRateB = 0.01)
    # main(EPISODE=400, VALIDATIONFREQ=10, RLALGORITHM="ActorCritic", PROJECTNAME="TESZTING_ACTORCRITIC")
    # #main(EPISODE=400, VALIDATIONFREQ=10, RLALGORITHM="ActorCritic", PROJECTNAME="TESZTING_ACTORCRITIC", LearningRateA = 0.003, LearningRateB = 0.01)
    # #main(EPISODE=400, VALIDATIONFREQ=10, RLALGORITHM="ActorCritic", PROJECTNAME="TESZTING_ACTORCRITIC", LearningRateA = 0.003, LearningRateB = 0.01)
    main()
    
    
    
    end = time.time()
    elapsed = end - start
    print(f'Time taken: {elapsed:.6f} seconds')