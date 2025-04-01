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
```
