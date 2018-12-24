import numpy as np

# Number of bins that you want to discretize the x space into
X_BINS = 12
# Number of bins that you want to discretize the y space into
Y_BINS = 12
# Number of velocities that you want to discretize the X velocity into
V_X = 2
# Number of velocities that you want to discretize the Y velocity into
V_Y = 3
# Number of locations that you want to discretize the Paddle location into
PADDLE_LOCATIONS = 12
# The number of actions allowed. 0 corresponds to moving down, 1 corresponds to not moving and 2 corresponds to moving up 
# DO NOT CHANGE THIS
NUM_ACTIONS = 3

def create_q_table():
	return np.zeros((X_BINS,Y_BINS,V_X,V_Y,PADDLE_LOCATIONS,NUM_ACTIONS))

def sanity_check(arr):
	if (type(arr) is np.ndarray and 
		arr.shape==(X_BINS,Y_BINS,V_X,V_Y,PADDLE_LOCATIONS,NUM_ACTIONS)): 
		return True
	else:
		return False

def save(filename,arr): 
	if sanity_check(arr):
		np.save(filename,arr)
		return True
	else:
		print("Failed to save model")
		return False

def load(filename):
	try:
		arr = np.load(filename)
		if sanity_check(arr):
			print("Loaded model succesfully")
			return arr
		print("Model loaded is not in the required format")
		return None
	except:
		print("Filename doesnt exist")
		return None