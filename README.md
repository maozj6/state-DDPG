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

All 3 methods calculate the smoothness of the z speed: 

**float32 w   	# Body z velocity (m/s)**

digit_number should be the position of Body z velocity (m/s) in the _next_state_

Method 1: Linear Trend Fitting

This method fits a linear function to the first 149 time steps of the selected velocity channel using least squares regression. It then predicts the 150th value and compares it to the actual value. The reward is computed based on the closeness between the predicted and actual value — the smaller the mean squared error, the higher the reward. This encourages the final value to follow the existing linear trend.

Method 2: Difference-Based Smoothness

This method calculates the average velocity change (delta) over the first 148 steps and compares it with the delta from step 149 to 150. The reward is higher when this final delta is close to the average delta, encouraging smooth transitions without abrupt spikes or drops in the velocity signal.

Method 3: High-Order Trend Smoothing (Savitzky-Golay Filter)

This method applies a Savitzky-Golay filter to smooth the first 149 velocity values, capturing higher-order polynomial trends. It then extrapolates the 150th value using the fitted polynomial and compares it with the actual value. A small deviation results in a high reward. This approach rewards consistency with the smooth and potentially non-linear trend in the preceding signal.
