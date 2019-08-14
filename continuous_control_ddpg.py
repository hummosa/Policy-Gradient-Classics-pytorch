
#%%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from unityagents import UnityEnvironment
env = UnityEnvironment(file_name='./../Reacher_Windows_x86_64/Reacher.exe')#, no_graphics=True)
# env = UnityEnvironment(file_name='./Reacher_One_Windows_x86_64/Reacher.exe')
# env = UnityEnvironment(file_name='/data/Reacher_One_Linux_NoVis/Reacher_One_Linux_NoVis.x86_64')
# env = UnityEnvironment(file_name= '/data/Reacher_Linux_NoVis/Reacher.x86_64')

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]

# Setting up hyperparameters
class Config():
    def __init__(self):
        self.state_size=  env_info.vector_observations.shape[1]
        self.action_size= brain.vector_action_space_size
        self.no_agents= len(env_info.agents)
        self.device= device
        self.gradient_clip = 1
        self.rollout_length = 1001
        self.episode_count = 251
        self.buffer_size = int(1e5)
        self.batch_size = 128
        self.lr_actor = 1e-4
        self.lr_critic = 1e-4
        self.discount_rate = 0.99           
        self.tau = 1e-3    
        self.weight_decay = 0
config = Config()


from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import Agent
agent = Agent(state_size=state_size, action_size=action_size, random_seed=1, config=config)

from tqdm import tqdm

# Defining the main training loop.
def ddpg(n_episodes=300, max_t=1000):
    scores_deque = deque(maxlen=100)
    scores = []
    max_score = -np.Inf
    
    for i_episode in tqdm(range(1, n_episodes+1)):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = np.zeros(config.no_agents)
        agent.reset()
        

        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done            
            
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            score += rewards
            
            if np.any(dones):
                break 
                
        
        scores_deque.append(score.mean())
        scores.append(score.mean())
        
        if i_episode % 10 == 0:
            # torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            # torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
    return scores



# Load available weights if needed to play a demo round
# agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
# agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

def play_round(env, brain_name, policy, config):
    env_info = env.reset(train_mode=False)[brain_name]    
    states = env_info.vector_observations                 
    scores = np.zeros(config.no_agents)                         
    while True:
        actions = policy(states, False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations         
        rewards = env_info.rewards                         
        dones = env_info.local_done                     
        scores += env_info.rewards                      
        states = next_states                               
        if np.any(dones):                                  
            break
    
    return np.mean(scores)
    
current_score = play_round(env, brain_name, agent.act, config)    
print('score this round: {}'.format(current_score))

# RUn the training loop
scores = ddpg(n_episodes=config.episode_count)

#Plot rewards
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


env.close()
