import gymnasium as gym
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.optim as optim
import random
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)

class dqn(nn.Module):
    def __init__(self, stateDims, nActions):
        super(dqn, self).__init__()
        #input layer, # observations -> 128 neurons (#x128 weight matrix & 128 bias vector)
        self.fc1 = nn.Linear(stateDims, 128)
        #hidden layer, 
        self.fc2 = nn.Linear(128, 128)
        #output layer, 
        self.fc3 = nn.Linear(128, nActions)

    def forward(self, x):
        x = nnf.relu(self.fc1(x))
        x = nnf.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class replayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, state, action, reward, nextState, done):
        experience = (state, action, reward, nextState, done)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batchSize):
        return random.sample(self.buffer, batchSize)
    
    def __len__(self):
        return len(self.buffer)

#training parameters
#number of states to train on at the same time
replayBatchSize = 64
#how many times lunar lander runs
nEpisodes = 10000
#discount for future rewards
g = 0.95
#exploratoin coefficient
e = 1.0
#final min exporation rate
eMin = 0.01
#explore decay rate
eDecay = 0.99999
#learning rate
lr = 0.001
#target update frequency
semester = 1000
#number inputs from state
stateDims = 8
#number possible actions per state
nActions = 4

#initialize neural net
episodePolicy = dqn(stateDims, nActions)


#load in trained policy
episodePolicy.load_state_dict(torch.load("lunarlanderDQN.pth"))


#second neural net for stability (target)
#landerSchool saved to gradSchool every semesterUpdate period
landmarkPolicy = dqn(stateDims, nActions)

#eval() is pytorch method: training mode -> eval mode
landmarkPolicy.eval()

#replay buffer size
replayStorage = replayBuffer(capacity = 50000)

#optimizer Adam updating weights and biases
#parameters() iterating over all weights & biases in this nn
optimizer = optim.Adam(episodePolicy.parameters(), lr)

#Mean Squared Loss function
#loss = mean(predicted_value - actual_value)^2
lossDiff = nn.MSELoss()

def chooseAction(state, epsilon):
    #random exploration rate
    if random.random() < epsilon:
        return random.randint(0, 3)
    #follow actual action values
    else:
        #FloatTensor(state) -> float tensor type array for torch
        #unsqueeze adding batch dimension for nn, (how many examples at once)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = episodePolicy(state_tensor)
        #highest q-value action (.item() convertin from tensor to regular number)
        return q_values.argmax().item()

def trainStep():
    batch = replayStorage.sample(replayBatchSize)
    states, actions, rewards, nextStates, dones = zip(*batch)

    #convert to pytorch tensors
    states = torch.FloatTensor(np.array(states))
    actions = torch.LongTensor(actions).unsqueeze(1)
    rewards = torch.FloatTensor(rewards).unsqueeze(1)
    nextStates = torch.FloatTensor(np.array(nextStates))
    dones = torch.FloatTensor(dones).unsqueeze(1)

    #get current q-value
    qValues = episodePolicy(states).gather(1, actions)

    #get max q-value for next state from landmark network
    nextQValue = landmarkPolicy(nextStates).max(1)[0].unsqueeze(1)

    #estimate new q-values
    estimateQValues = rewards + (g * nextQValue * (1 - dones))

    #loss
    loss = lossDiff(qValues, estimateQValues)

    #optimize model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#test parameters
nEpisodes = 10
e = 0
eMin = 0
episodePolicy.eval()


#accessing prebuilt lunar lander environment from gymnasium
env = gym.make("LunarLander-v3", render_mode='human')

rewards = []

for episode in range(nEpisodes):
    state, info = env.reset()

    #print(f"start state x {state[0]}, y {state[1]}")

    done = False
    totalReward = 0
    frameCounter = 0

    while not done:
        action= chooseAction(state, e)
        #action = env.action_space.sample()
        nextState, reward, done, trunc, info = env.step(action)

        replayStorage.add(state, action, reward, nextState, done)
        if len(replayStorage) >= replayBatchSize:
            trainStep()

        state = nextState

        totalReward += reward

        if state[6] == 1 and state[7] == 1:
            done = True

    e = max(eMin, e*eDecay)

    if episode % semester == 0:
        #state_dict() -> pytorch dictionary storing all model's weights and biases
        #load_state_dict copies into new dqn
        landmarkPolicy.load_state_dict(episodePolicy.state_dict())

    rewards.append(totalReward)
    if episode % 1 == 0:
        print(f"Episode {episode}: reward = {totalReward:.2f}, explore rate {e}")
    
    #print(f"x-coor {state[0]}")
    #print(f"y-coor {state[1]}")
    #print(f"x-vel {state[2]}")
    #print(f"y-vel {state[3]}")
    #print(f"angle {state[4]}")
    #print(f"ang vel {state[5]}")
    #print(f"leg 1 {state[6]}")
    #print(f"leg-2 {state[7]}")
    
torch.save(episodePolicy.state_dict(), "lunarlanderDQN.pth")

env.close()

#smoothing line function
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

#calculate smoothed rewards
smoothed_rewards = moving_average(rewards, window_size=100)

#create plot
plt.figure(figsize=(12,6))
plt.plot(rewards, color='lightblue', label='Raw Rewards')
plt.plot(range(len(smoothed_rewards)), smoothed_rewards, color='red', linewidth=2, label='Smoothed Rewards (100 episodes)')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Lunar Lander DQN Training Progress')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('training_rewards.png')
plt.show()