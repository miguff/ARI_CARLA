from Environment import EnvironmentClass
from RLAlgorithm import ActorCriticAgent
from datetime import datetime
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import time

def main(LOGDIR = "logs", PROJECTNAME = "ReinforcementLearningAlgo", RLALGORITHM = "ActorCritic", EPISODE = 100, VALIDATIONFREQ= 5):

    runname = f'{RLALGORITHM}_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=f'./{LOGDIR}/{PROJECTNAME}/{runname}')


    carenv = EnvironmentClass("Training")
    INPUTDIM = len(carenv.objectreturn)
    EPISODE = EPISODE
    agent = ActorCriticAgent(alpha=0.000005, beta = 0.0001, input_dims=[INPUTDIM], gamma=0.99, layer1_size=256, layer2_size=256, writer=writer)
    REWARDS = []
    TRAINING_STEP = 0
    VALIDATION_STEP = 0

    for episode in range(EPISODE):
        if episode % VALIDATIONFREQ == 0 and episode != 0:
            print("Validation Episode")
        else:
            print("Training Episode")
        #Setup Initial State -> Here only just the done and terminated parameter
        objectssata, reward, done, terminated, next_step = carenv.reset()
        EPISODE_TOTAL_REWARD = 0

        #Make a variable to check whether the episode ended or not
        REACHED_GOAL = False
        while REACHED_GOAL != True:
            
            #Check if the model needs to make a prediction, or use the PID value
            if next_step == 1:
                action = agent.choose_action(objectssata)
                BASE = False
                returnvalues = carenv.step(action)
            else:
                BASE = True
                returnvalues = carenv.step()
            #This Branch is for when the car used the PID Controller, insted of the RL action.
            #It is needed, because, We do not want to train on data, that was not used during simulation.
            if BASE:
                objectssata, reward, done, terminated, next_step = returnvalues
                if done == True:
                    REACHED_GOAL = True
                continue
            #print(carenv)
            objectssata_, reward, done, terminated, next_step = returnvalues
            EPISODE_TOTAL_REWARD += reward
            #If we will be here, that is where our modell will learn, here will be defined the learning sequence.
            

            if episode % VALIDATIONFREQ == 0:
                writer.add_scalar("Validation Step Reward", reward, VALIDATION_STEP)
                VALIDATION_STEP += 1
            else:
                agent.learn(objectssata, reward, objectssata_, done, TRAINING_STEP)
                writer.add_scalar("Training Step Reward", reward, TRAINING_STEP)
                TRAINING_STEP += 1
            writer.flush()
            
            objectssata = objectssata_

            if done == True:
                REACHED_GOAL = True
            
        print(f"Episode {episode}, Reward: {EPISODE_TOTAL_REWARD:.2f}")
        if episode % VALIDATIONFREQ == 0:
            writer.add_scalar("Validation Episode Reward", EPISODE_TOTAL_REWARD, episode)
        else:
            writer.add_scalar("Training Episode Reward", EPISODE_TOTAL_REWARD, episode)
        
        writer.flush()
        REWARDS.append(EPISODE_TOTAL_REWARD)

    carenv.cleanup()
    writer.close()
    plt.plot(REWARDS)
    plt.show()
    


if __name__ == "__main__":
    start = time.time()
    main(EPISODE=200, VALIDATIONFREQ=40)
    end = time.time()
    elapsed = end - start
    print(f'Time taken: {elapsed:.6f} seconds')