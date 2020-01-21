
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("MountainCar-v0")

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING  = EPISODES // 2
decay_value = epsilon/ (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


discrete_OS_SIZE = [20] * len(env.observation_space.high)
discrete_OS_winn_SIZE = (env.observation_space.high - env.observation_space.low)/discrete_OS_SIZE

q_table = np.random.uniform(low=-2,high=0, size=(discrete_OS_SIZE + [env.action_space.n]))

ep_rewards = []
aggr_ep_reward = {'ep': [], 'avg': [], 'min' : [], 'max ': []}



def get_discrete(state):
    discrete_size = (state - env.observation_space.low) / discrete_OS_winn_SIZE
    return tuple(discrete_size.astype(np.int))



for ep in range(EPISODES):
    ep_reward = 0
    if ep % SHOW_EVERY == 0:
        print(ep)
        render = True
    else:
        render = False


    discrete_state = get_discrete(env.reset())






    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        
        new_state, reward, done, _ = env.step(action)
        ep_reward += reward

        new_discrete_state = get_discrete(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state+(action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"we made it on {ep}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
    if END_EPSILON_DECAYING >= ep >= START_EPSILON_DECAYING:
        epsilon -= decay_value

    ep_rewards.append(ep_reward)

    if not ep % SHOW_EVERY:
        avarege_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_reward['ep'].append(ep)
        aggr_ep_reward['avg'].append(avarege_reward)
        aggr_ep_reward['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_reward['max '].append(max(ep_rewards[-SHOW_EVERY:]))

        print(f'episode{ep} avg {avarege_reward} min {min(ep_rewards[-SHOW_EVERY:])} max {max(ep_rewards[-SHOW_EVERY:])}')




env.close()
plt.plot(aggr_ep_reward['ep'],aggr_ep_reward['avg'], label='avg')
plt.plot(aggr_ep_reward['ep'],aggr_ep_reward['min'], label='min')
plt.plot(aggr_ep_reward['ep'],aggr_ep_reward['max '], label='max')
plt.legend(loc=4)
plt.show()








