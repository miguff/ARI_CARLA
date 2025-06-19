
from RLAlgorithm import ActorCriticAgent, DDPGAgent
from Environment import EnvironmentClass
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def Learn(agent: DDPGAgent, carenv: EnvironmentClass, writer: SummaryWriter, VALIDATIONFREQ=5, EPISODE=100, ):

    EPISODE = EPISODE
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
            
            if any(np.isnan(v) for v in objectssata):
                print("NaN detected in observation: ", objectssata)
                REACHED_GOAL = True
                continue

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
                objectssata_, reward, done, terminated, next_step = returnvalues
                if done == True:
                    REACHED_GOAL = True
                continue
            #print(carenv)
            objectssata_, reward, done, terminated, next_step = returnvalues
            EPISODE_TOTAL_REWARD += reward
            #If we will be here, that is where our modell will learn, here will be defined the learning sequence.
            

            if episode % VALIDATIONFREQ == 0 and episode != 0:
                writer.add_scalar("Validation Step Reward", reward, VALIDATION_STEP)
                VALIDATION_STEP += 1
            else:
                print("Action")
                print(action)
                agent.learn(objectssata, action, reward, objectssata_, done, TRAINING_STEP)
                print("Írom bele az értékeket")
                print(reward)
                print(TRAINING_STEP)
                writer.add_scalar("Training Step Reward", reward, TRAINING_STEP)
                TRAINING_STEP += 1
            writer.flush()
            
            objectssata = objectssata_

            if done == True:
                REACHED_GOAL = True
            
        print(f"Episode {episode}, Reward: {EPISODE_TOTAL_REWARD:.2f}")
        if episode % VALIDATIONFREQ == 0 and episode != 0:
            writer.add_scalar("Validation Episode Reward", EPISODE_TOTAL_REWARD, episode)
            agent.save_models(str(EPISODE_TOTAL_REWARD))
        else:
            print(EPISODE_TOTAL_REWARD)
            writer.add_scalar("Training Episode Reward", EPISODE_TOTAL_REWARD, episode)
        
        writer.flush()
        REWARDS.append(EPISODE_TOTAL_REWARD)

    carenv.cleanup()
    writer.close()
