import gymnasium as gym
import numpy as np

env = gym.make("MountainCar-v0", render_mode="human")
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25_000
SHOW_EVERY = 1000

# breaking continous data into discrete data
DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/ DISCRETE_OS_SIZE

# Starting with random Q table of size/shape = [20,20,3]
q_table  = np.random.uniform(low=-2, high = 0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))

for episode in range(EPISODES):
    # env.reset() returns two values -> initial state, info 
    discrete_state = get_discrete_state(env.reset()[0])
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:
        # print(episode)
        action = np.argmax(q_table[discrete_state])
        new_state, reward, terminated, truncated, info = env.step(action)

        if truncated or terminated:
            done = True        

        if episode % SHOW_EVERY == 0:
            env.render()
        if not done:
            # if not done -> update q table
            new_discrete_state = get_discrete_state(new_state)
            max_future_q = np.max(q_table[new_discrete_state])

            #current Q value

            current_q = q_table[new_discrete_state + (action, )]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action, )] = new_q

        # if reached goal
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action, )] = 0


        

    env.close()
