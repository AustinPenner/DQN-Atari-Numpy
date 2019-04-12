import h5py
import random
import numpy as np
import os


class Experience(object):
	# Store and retrieve Atari states for experience replay
	def __init__(self, maxlen):
		self.maxlen = maxlen

		self.frames = Memory(maxlen)
		self.actions = Memory(maxlen)
		self.rewards = Memory(maxlen)
		self.initials = Memory(maxlen)
		self.terminals = Memory(maxlen)

	def add(self, frame, action, reward, initial, terminal):
		# Add state to memory
		self.frames.append(frame)
		self.actions.append(action)
		self.rewards.append(reward)
		self.initials.append(initial)
		self.terminals.append(terminal)

	def length(self):
		return len(self.actions)

	def get(self, start, end):
		# Get states directly from Memory
		get_actions, get_rewards, get_terminals, get_initials, get_frames = [], [], [], [], []
		for i in range(start, end):
			get_actions.append(self.actions[i])
			get_rewards.append(self.rewards[i])
			get_terminals.append(self.terminals[i])
			get_initials.append(self.initials[i])
			get_frames.append(self.frames[i])

		return (get_actions, get_rewards, get_terminals, get_initials, get_frames)

	def sample(self, batch_size):
		# Get minibatch of states
		n_states = len(self.actions)
		minibatch = random.sample(xrange(n_states), batch_size)

		mb_frames_prev = np.zeros((batch_size, 84, 84, 4))
		mb_frames_next = np.zeros((batch_size, 84, 84, 4))
		mb_actions, mb_rewards, mb_terminals = [], [], []

		for mb_idx, state_idx in enumerate(minibatch):
			mb_actions.append(self.actions[state_idx])
			mb_rewards.append(self.rewards[state_idx])
			mb_terminals.append(self.terminals[state_idx])

			# Get the 4 most recent frames and the following one in order to
			# update the model.
			mb_frames = [self.frames[(state_idx - 3) % n_states], 
				self.frames[(state_idx - 2) % n_states], 
				self.frames[(state_idx - 1) % n_states], 
				self.frames[state_idx], 
				self.frames[(state_idx + 1) % n_states]]

			# If the episode starts here set previous frames to zero.
			initial_state = False
			for i in range(3, -1, -1):
				initial_idx = (state_idx + i - 3) % n_states
				initial = self.initials[initial_idx]

				if initial_state == True:
					mb_frames[i] = np.zeros((84, 84))
				elif initial == True:
					initial_state = True

			mb_frames_prev[mb_idx, :, :, :] = np.stack(mb_frames[0:4], axis=-1)
			mb_frames_next[mb_idx, :, :, :] = np.stack(mb_frames[1:5], axis=-1)

		return (mb_frames_prev, mb_frames_next, mb_actions, mb_rewards, mb_terminals)

	def save(self, filename = None):
		# Save experience with h5py
		if filename == None:
			filename = 'experience.h5'
		n_states = len(self.actions)
		
		with h5py.File(filename, 'w') as f:
			f.create_dataset('Frames', data=self.frames[0:n_states])
			f.create_dataset('Actions', data=self.actions[0:n_states])
			f.create_dataset('Rewards', data=self.rewards[0:n_states])
			f.create_dataset('Initials', data=self.initials[0:n_states])
			f.create_dataset('Terminals', data=self.terminals[0:n_states])
			f.create_dataset('Index', data=self.actions.cur)

	def load(self, filename = None):
		# Load experience with h5py
		if filename == None:
			filename = 'experience.h5'

		if os.path.isfile(filename):
			with h5py.File(filename, 'r') as f:
				h5_frames = np.array(f['Frames'])
				h5_actions = np.array(f['Actions'])
				h5_rewards = np.array(f['Rewards'])
				h5_initials = np.array(f['Initials'])
				h5_terminals = np.array(f['Terminals'])
				idx = np.array(f['Index'])

			for i in range(len(h5_actions)):
				self.add(h5_frames[i], 
					h5_actions[i], 
					h5_rewards[i], 
					h5_initials[i], 
					h5_terminals[i])

			self.frames.cur = idx
			self.actions.cur = idx
			self.rewards.cur = idx
			self.initials.cur = idx
			self.terminals.cur = idx


class Memory(object):
	# Ring buffer class to hold game state information. This is
	# used because deque is very slow.
	def __init__(self, maxlen):
		self.maxlen = maxlen
		self.data = [None] * self.maxlen
		self.cur = 0
		self.full = False

	def append(self, x):
		self.data[self.cur] = x
		self.cur = (self.cur + 1) % self.maxlen
		if self.cur == self.maxlen - 1:
			self.full = True

	def __len__(self):
		if self.full:
			return self.maxlen
		else:
			return self.cur

	def __getitem__(self, idx):
		return self.data[idx]