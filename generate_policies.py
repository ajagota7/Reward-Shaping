import numpy as np 
from policy_functions import manhattan_distance
from agent import Agent

def create_policy_set(env, policy_func, policy_map, num_episodes):
    # Create a list to store policies as trajectories
    policies = []

    # Run multiple episodes
    for episode in range(num_episodes):
        # Create a new Agent for each episode to generate a different policy
        agent = Agent(epsilon=0.0)

        # Run an episode
        env.reset()
        done = False
        trajectory = []  # Store the trajectory for the current episode
        cumulative_reward = 0.0  # Initialize cumulative reward
        timestep = 0  # Initialize timestep counter
        while not done:
            state = env.agent_position  # Get the current state
            action = agent.select_action(lambda: policy_func(state, policy_map))
            next_state, reward, done = env.step(action)
            dist_to_terminal = 0.8 * manhattan_distance(state, env.end)

            # Compute cumulative reward
            cumulative_reward += reward

            # Store the (timestep, state, action, reward, next_state, cumulative_reward, dist_to_terminal) tuple in the trajectory
            # trajectory.append((state, action, reward, next_state, cumulative_reward, dist_to_terminal, timestep))
            # state_tensor = torch.tensor(state, dtype=torch.float32)

            # Store the (timestep, state, action, reward, next_state, cumulative_reward, dist_to_terminal) tuple in the trajectory
            step_data = np.array([state, action, reward, next_state, cumulative_reward, dist_to_terminal, timestep])
            # step_data = np.array([state_tensor, action, reward, next_state, cumulative_reward, dist_to_terminal, timestep])

            trajectory.append(step_data)

            timestep += 1  # Increment timestep counter

        # Append the trajectory to the policies list
        policies.append(trajectory)

    return policies
