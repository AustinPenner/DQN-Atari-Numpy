import numpy as np
import h5py
import os
from numba import jit
import random


class DQN_Model(object):
	def __init__(self, n_actions, alpha, discount, beta, rmsprop_eps):
		self.n_actions = n_actions
		self.discount = discount
		self.alpha = alpha
		self.beta = beta
		self.rmsprop_eps = rmsprop_eps

	def initialize_model(self):
		# Xavier weight initialization
		self.W1 = np.random.randn(8,8,4,16) / np.sqrt(84*84*4)
		self.W2 = np.random.randn(4,4,16,32) / np.sqrt(20*20*16)
		self.W3 = np.random.randn(9*9*32,256) / np.sqrt(9*9*32)
		self.W4 = np.random.randn(256, self.n_actions) / np.sqrt(256)

		# RMSprop initialization
		self.S_W1 = np.zeros(self.W1.shape)
		self.S_W2 = np.zeros(self.W2.shape)
		self.S_W3 = np.zeros(self.W3.shape)
		self.S_W4 = np.zeros(self.W4.shape)

	def forward_prop(self, X):
		# Map input frames from 0-255 to 0-1
		X = X.astype('float32')
		X = X / 255.0

		# Do forward pass through the network
		A1, cache1 = self.conv_forward(X, self.W1, 4, activation='relu')
		A2, cache2 = self.conv_forward(A1, self.W2, 2, activation='relu')
		A3, cache3 = self.fc_forward(A2, self.W3, activation='relu')
		A4, cache4 = self.fc_forward(A3, self.W4, activation='linear')

		caches = [cache1, cache2, cache3, cache4]

		return A4, caches

	@jit
	def conv_forward(self, A_prev, W, stride, activation):
		# Do forward pass on convolution layer
		(m, h_prev, w_prev, n_c_prev) = A_prev.shape
		(f, f, n_c_prev, n_c) = W.shape

		# Compute dimensions of output volume
		n_h = int((h_prev - f)/stride + 1)
		n_w = int((w_prev - f)/stride + 1)
		Z = np.zeros((m, n_h, n_w, n_c))

		for i in range(m):
			for h in range(n_h):
				for w in range(n_w):
					for c in range(n_c):
						# Get convolution boundaries of slice from A_prev
						lower_w = w * stride
						upper_w = w * stride + f
						lower_h = h * stride
						upper_h = h * stride + f

						# Convolve and store in Z
						a_slice_prev = A_prev[i, lower_h:upper_h, lower_w:upper_w, :]
						convolve_slice = np.multiply(a_slice_prev, W[:, :, :, c])
						Z[i, h, w, c] = np.sum(convolve_slice)

		# Use relu activation function
		A = self.activate(Z, activation)

		cache = (A_prev, Z)
		return A, cache

	@jit
	def fc_forward(self, A_prev, W, activation):
		# Do forward pass on fully connected layer
		shape = A_prev.shape
		if len(shape) > 2:
			A_prev = np.reshape(A_prev, (shape[0], np.prod(shape[1:])))

		Z = np.matmul(A_prev, W)

		# Use relu or linear activation
		A = self.activate(Z, activation)

		cache = (A_prev, Z)
		return A, cache

	def update(self, minibatch):
		# Perform gradient descent update with minibatch
		(observations_prev, observations, actions, rewards, terminals) = minibatch
		m = len(actions)

		# Retrieve q values and reward for the loss function L
		# grad(L) = Q_prev - (r + discount*Q_next)
		# Set Q_next to 0 if this was a terminal state
		qvalues, _ = self.forward_prop(observations)
		qvalues = np.amax(qvalues, axis=1)
		qvalues = qvalues * ~np.array(terminals)
		targets = rewards + self.discount*qvalues

		# Get prev q values and calculate grad(L)
		qvalues_prev, caches_prev = self.forward_prop(observations_prev)
		qvalues_prev = qvalues_prev[np.arange(m), actions]
		losses = np.zeros((m, self.n_actions))
		losses[np.arange(m), actions] = qvalues_prev - targets

		# Use backprop to find gradients
		dW1, dW2, dW3, dW4 = self.backward_prop(caches_prev, losses)

		# Use gradients to update model with RMSprop
		self.W1, self.S_W1 = self.rmsprop(self.W1, dW1, self.S_W1, self.alpha)
		self.W2, self.S_W2 = self.rmsprop(self.W2, dW2, self.S_W2, self.alpha)
		self.W3, self.S_W3 = self.rmsprop(self.W3, dW3, self.S_W3, self.alpha)
		self.W4, self.S_W4 = self.rmsprop(self.W4, dW4, self.S_W4, self.alpha)

	def backward_prop(self, caches, losses):
		# Compute gradients for weight matrices
		m = losses.shape[0]
		(X, Z1) = caches[0]
		(A1, Z2) = caches[1]
		(A2, Z3) = caches[2]
		(A3, Z4) = caches[3]

		dA4 = losses
		dW4, dA3 = self.fc_backward(dA4, Z4, A3, self.W4, activation = 'linear')
		dW3, dA2 = self.fc_backward(dA3, Z3, A2, self.W3, activation = 'relu')
		dW2, dA1 = self.conv_backward(dA2, Z2, A1, self.W2, 2, activation = 'relu', islayer1=False)
		dW1, _ = self.conv_backward(dA1, Z1, X, self.W1, 4, activation = 'relu', islayer1=True)

		return dW1, dW2, dW3, dW4

	@jit
	def fc_backward(self, dA, Z, A_prev, W, activation):
		m = dA.shape[0]

		# Backward prop on activation
		dZ = self.activate_backward(dA, Z, activation)

		# Backward prop on fully connected layer
		dW = 1.0/m*np.dot(A_prev.T, dZ)
		dA_prev = np.dot(dZ, W.T)

		return dW, dA_prev

	@jit
	def conv_backward(self, dA, Z, A_prev, W, stride, activation, islayer1):
		# Backward prop on convolutional layer
		(m, h_prev, w_prev, n_c_prev) = A_prev.shape
		(f, f, n_c_prev, n_c) = W.shape

		n_h = int((h_prev - f)/stride + 1)
		n_w = int((w_prev - f)/stride + 1)
		shape = dA.shape
		if len(shape) < 4:
			dA = np.reshape(dA, (m, n_h, n_w, n_c))

		# Backward prop on activation
		dZ = self.activate_backward(dA, Z, activation)
		dW = np.zeros(W.shape)
		dA_prev = np.zeros(A_prev.shape)

		for i in range(m):
			for h in range(n_h):
				for w in range(n_w):
					for c in range(n_c):
						# Get convolution boundaries from A_prev
						lower_w = w * stride
						upper_w = w * stride + f
						lower_h = h * stride
						upper_h = h * stride + f

						# Update dW with slice from A_prev and dZ
						a_slice_prev = A_prev[i, lower_h:upper_h, lower_w:upper_w, :]
						z = dZ[i, h, w, c]

						dW[:, :, :, c] += z*a_slice_prev

						# The gradient of A_prev is unnecessary on first layer because A_prev
						# is just the input to the CNN.
						if not islayer1:
							dA_prev[i, lower_h:upper_h, lower_w:upper_w, :] += W[:, :, :, c]*z

		dW = 1.0/m*dW
		return dW, dA_prev

	def rmsprop(self, Wi, dWi, S_Wi, alpha):
		# Use RMSprop optimizer
		S_Wi = self.beta*S_Wi + (1-self.beta)*np.square(dWi)
		Wi += -alpha*np.divide(dWi, np.sqrt(S_Wi + self.rmsprop_eps))

		return Wi, S_Wi

	def activate(self, Z, activation):
		# Activation function for forward pass
		if activation == 'relu':
			A = np.maximum(Z, 0)
		else:
			A = Z

		return A

	def activate_backward(self, dA, Z, activation):
		# Backprop through activation function
		if activation == 'relu':
			dZ = np.copy(dA)
			dZ[Z < 0] = 0
		else:
			dZ = dA

		return dZ

	def save_model(self, filename = None):
		# Save model using h5py
		if filename == None:
			filename = 'weights.h5'

		with h5py.File(filename, 'w') as f:
			f.create_dataset('W1', data=self.W1)
			f.create_dataset('W2', data=self.W2)
			f.create_dataset('W3', data=self.W3)
			f.create_dataset('W4', data=self.W4)
			f.create_dataset('S_W1', data=self.S_W1)
			f.create_dataset('S_W2', data=self.S_W2)
			f.create_dataset('S_W3', data=self.S_W3)
			f.create_dataset('S_W4', data=self.S_W4)

	def load_model(self, filename = None):
		# Load model using h5py, if this file exists. Else, initialize a new model.
		if filename == None:
			filename = 'weights.h5'

		if os.path.isfile(filename):
			with h5py.File(filename, 'r') as f:
				self.W1 = np.array(f['W1'])
				self.W2 = np.array(f['W2'])
				self.W3 = np.array(f['W3'])
				self.W4 = np.array(f['W4'])
				self.S_W1 = np.array(f['S_W1'])
				self.S_W2 = np.array(f['S_W2'])
				self.S_W3 = np.array(f['S_W3'])
				self.S_W4 = np.array(f['S_W4'])
		else:
			self.initialize_model()