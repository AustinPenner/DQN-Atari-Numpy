import gym
import numpy as np
from PIL import Image
from dqn import DQN_Model
from atari_experience import Experience
import time


def preprocess(observation):
	# Downsample image to 84x84 grayscale
	img = Image.fromarray(observation)
	img = img.resize((84,110))
	img = img.crop((0,26,84,110))
	img = img.convert('L')

	return np.array(img, dtype='uint8')

def clip_reward(reward):
	# Set rewards to -1, 0, or 1
	if reward > 0:
		reward = 1
	elif reward < 0:
		reward = -1
	else:
		reward = 0
	return reward

def take_action(model, observations, epsilon):
	# Take random action with probability 1-epsilon. Else follow epsilon-greedy strategy.
	observations = np.reshape(np.stack(observations, axis=-1), (1, 84, 84, 4))
	if np.random.uniform() < epsilon:
		action = env.action_space.sample()
	else:
		actions, _ = model.forward_prop(observations)
		action = np.argmax(actions)

	next_obs, reward, terminal, info = env.step(action)
	reward = clip_reward(reward)

	return action, reward, next_obs, terminal

def train_atari(env, model, memory, training_steps, init_epsilon, final_epsilon, minibatch_size, 
				rendergame, updatemodel=False):
	# Train a Deep Q Network to play Atari
	total_steps = 0

	while total_steps < training_steps:
		episode_reward = 0
		initialstep = True
		terminalstep = False
		
		# We input the 4 most recent frames to the DQN. During the first 4 steps in game
		# we use zeros for these empty frames.
		ep_start = np.zeros((84, 84), dtype='uint8')
		observations = [ep_start] * 4
		obs = env.reset()

		while not terminalstep:
			if total_steps % 1000 == 0:
				print "step " + str(total_steps)

			if rendergame:
				env.render()
				time.sleep(0.05)

			# Use epsilon greedy policy to choose actions. Epsilon is linearly annealed over
			# total number of training steps.
			epsilon = init_epsilon - (total_steps/float(training_steps))*(init_epsilon - final_epsilon)
			action, reward, obs, terminalstep = take_action(model, observations, epsilon)

			# Downsample image and stack with 3 prior frames
			obs = preprocess(obs)
			observations.append(obs)
			observations.pop(0)

			# Store states in memory
			memory.add(obs, action, reward, initialstep, terminalstep)

			# Use experience replay to update the neural network
			if updatemodel == True:
				minibatch = memory.sample(minibatch_size)
				model.update(minibatch)

			initialstep = False
			episode_reward += reward
			total_steps += 1

			if total_steps >= training_steps:
				break

	if not rendergame:
		model.save_model()
		memory.save()


# DQN hyperparameters
MINIBATCH_SIZE = 32
ALPHA = 0.0002
DISCOUNT = 0.99
RMSPROP_BETA = 0.95
RMSPROP_EPSILON = 0.01

TRAINING_STEPS = 100000
MAX_STATES_STORED = 100000
INIT_EPSILON = 1.0
FINAL_EPSILON = 0.1
ENV_NAME = 'BreakoutDeterministic-v4'

rendergame = False

if __name__ == "__main__":
	env = gym.make(ENV_NAME)
	n_actions = env.action_space.n
	model = DQN_Model(n_actions, ALPHA, DISCOUNT, RMSPROP_BETA, RMSPROP_EPSILON)
	model.load_model()
	memory = Experience(MAX_STATES_STORED)
	memory.load()
	newmodel = (memory.length() == 0)

	if not rendergame:
		if newmodel:
			# First, need Atari games to pull in for backprop. Initially play Atari randomly
			# to populate examples for experience replay
			train_atari(env, model, memory, MAX_STATES_STORED, 1.0, 
						1.0, MINIBATCH_SIZE, rendergame, updatemodel=False)

		# Train DQN on Atari. Epsilon is linearly annealed over (TRAINING_STEPS) steps
		train_atari(env, model, memory, TRAINING_STEPS, INIT_EPSILON, 
					FINAL_EPSILON, MINIBATCH_SIZE, rendergame, updatemodel=True)

	else:
		# Render game and save game recording
		env = gym.wrappers.Monitor(env, "recording", force=True)
		train_atari(env, model, memory, TRAINING_STEPS, INIT_EPSILON, 
					FINAL_EPSILON, MINIBATCH_SIZE, rendergame, updatemodel=False)