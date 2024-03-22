import gymnasium as gym
import numpy as np
from mdp import *


if __name__ == '__main__':
    env = gym.make("Taxi-v3")

    # discount factors
    discount_factors = [0.5, 0.7, 0.9]

    # number of actions and states
    num_actions = env.action_space.n
    num_states = env.observation_space.n

    env.reset()

    env.render()

    for discount_factor in discount_factors:

        # find the best policy
        policy, _ = value_iteration(env, num_actions, num_states, discount_factor=discount_factor)
        # policy, _ = policy_iteration(env, num_actions, num_states, discount_factor=discount_factor)
        average_steps = []
        average_rewards = []
        # test the policy
        for num_iterations in [50, 100]:
            steps_total = []
            rewards_total = []

            for _ in range(num_iterations):
                state, _ = env.reset()
                steps = 0
                total_reward = 0

                while True:
                    action = np.argmax(policy[state])
                    state, reward, done, _, _ = env.step(action)
                    # env.render()
                    steps += 1
                    total_reward += reward

                    if done:
                        steps_total.append(steps)
                        rewards_total.append(total_reward)
                        break

            avg_steps = np.mean(steps_total)
            average_steps.append(avg_steps)
            avg_reward = np.mean(rewards_total)
            average_rewards.append(avg_reward)

            # print(f"Discount Factor: {discount_factor}, Iterations: {num_iterations}")
            # print(f"Average steps to goal: {avg_steps}")
            # print(f"Average reward: {avg_reward}\n")

        print(f"Average steps with factor {discount_factor}: {np.mean(average_steps)}")
        print(f"Average reward with factor {discount_factor}: {np.mean(average_rewards)}\n")
    env.close()

