# DQN Atari in Numpy
This is an implementation of "Playing Atari with Deep Reinforcement Learning" (2013) using NumPy, without Tensorflow. The neural network forward and backward pass were made with NumPy. OpenAI's Gym runs the Atari emulator. Atari game states and the DQN model are saved as HDF5 files with h5py.

## Dependencies
All code was written in Python 2.7. Used [OpenAI Gym](https://github.com/openai/gym) for the Atari emulator. Required packages are NumPy, h5py, Pillow, and numba.

## Installing
git clone https://github.com/AustinPenner/DQN-Atari-Numpy.git

## Usage
Navigate to the directory and run with Python 2.7
```
cd DQNAtariNumpy
python learn_atari.py
```
Running this command will start training the Deep Q Network on Breakout. 

An epsilon-greedy strategy was used, which defaults to a linearly annealed epsilon from 1 to 0.1 over all training steps. These values can be modified in learn_atari.py along with all other hyperparameters.

## Reference
1. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)
2. [OpenAI Gym](https://github.com/openai/gym)