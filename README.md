# state-DDPG


TODO:

1. Train the LSTM world model


2. Substitute the code of LSTM， change it to your own LSTM code

```python
class LSTMModel(nn.Module):
```


3. Check the state and controller-action size :

```python
STATE_DIM = 17
ACTION_DIM = 5
```

4. Load the well-trained saved parameters of the LSTM model

```python
    env = LSTMModel()
```




5. Choose one reward functions:

```python
            reward = reward_function(next_state,digit_number)

def compute_reward(data, digit_number):

def compute_reward_diff(data, digit_number):


def compute_reward_savgol(data, digit_number, window_length=21, polyorder=3):

```

1. replace the LSTM model to Sujan's model and load the parameter  ( I don't know his input format,

in RL, state should be like (1-step,17)

but in LSTM, it maybe several step input (20-step,17). For the initialization, load the data, ask **Sujan** for the training and test data.

2. change the reward function, (let the airplane to go down, which means lower altitude/height, higher reward) 

find the z_position in the state and make a new reward function.

ask Anuj, which one is position_z



std_msgs/Header header
	builtin_interfaces/Time stamp
    	int32 sec
    	uint32 nanosec
	string frame_id

# Original States
# @warning roll, pitch and yaw have always to be valid, the quaternion is optional
float32[3] position	# north, east, down (m)
float32 va    	# Airspeed (m/s)
float32 alpha	# Angle of attack (rad)
float32 beta	# Slide slip angle (rad)
float32 phi    	# Roll angle (rad)
float32 theta   # Pitch angle (rad)
float32 psi    	# Yaw angle (rad)
float32 chi    	# Course angle (rad)
float32 u   	# Body x velocity (m/s)
float32 v   	# Body y velocity (m/s)
float32 w   	# Body z velocity (m/s)
float32 p    	# Body frame rollrate (rad/s)
float32 q    	# Body frame pitchrate (rad/s)
float32 r    	# Body frame yawrate (rad/s)
float32 vg    	# Groundspeed (m/s)
float32 wn    	# Winorth component (m/s)
float32 we    	# Windspeed east component (m/s)

# Additional States for convenience
float32[4] quat    	# Quaternion (wxyz, NED)
bool quat_valid    	# Quaternion valid
float32 chi_deg    	# Wrapped course angle (deg)
float32 psi_deg    	# Wrapped yaw angle (deg)
float32 initial_lat 	# Initial/origin latitude (lat. deg)
float32 initial_lon 	# Initial/origin longitude (lon. deg)
float32 initial_alt 	# Initial/origin altitude (m)


All 3 methods calculate the **smoothness** of the z speed: 


**float32 w   	# Body z velocity (m/s)**

digit_number should be the position of Body z velocity (m/s) in the _next_state_

Method 1: Linear Trend Fitting

This method fits a linear function to the first 149 time steps of the selected velocity channel using least squares regression. It then predicts the 150th value and compares it to the actual value. The reward is computed based on the closeness between the predicted and actual value — the smaller the mean squared error, the higher the reward. This encourages the final value to follow the existing linear trend.

Method 2: Difference-Based Smoothness

This method calculates the average velocity change (delta) over the first 148 steps and compares it with the delta from step 149 to 150. The reward is higher when this final delta is close to the average delta, encouraging smooth transitions without abrupt spikes or drops in the velocity signal.

Method 3: High-Order Trend Smoothing (Savitzky-Golay Filter)

This method applies a Savitzky-Golay filter to smooth the first 149 velocity values, capturing higher-order polynomial trends. It then extrapolates the 150th value using the fitted polynomial and compares it with the actual value. A small deviation results in a high reward. This approach rewards consistency with the smooth and potentially non-linear trend in the preceding signal.


![image](https://github.com/user-attachments/assets/97e0ead8-d2e9-49b8-b34f-b1f365a372a6)
