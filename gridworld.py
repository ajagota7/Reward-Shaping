import numpy as np

class GridWorld:
    def __init__(self, height, width, start, end, bad_region_clusters, good_region_clusters, final_reward, sparsity):
        self.height = height
        self.width = width
        self.start = start
        self.end = end
        self.bad_region_clusters = bad_region_clusters
        self.good_region_clusters = good_region_clusters
        # self.good_region_reward = good_region_reward
        # self.bad_region_reward = bad_region_reward
        self.final_reward = final_reward
        self.sparsity = sparsity

        self.state_rewards = self.generate_state_rewards()
        self.reset()

    def reset(self):
        self.agent_position = self.start


    def generate_state_rewards(self):
        state_rewards = {}
        for x in range(self.width):
            for y in range(self.height):
                state_reward = 0.0

                if (x, y) == self.start:
                    state_reward = 0.0
                elif (x, y) == self.end:
                    state_reward = self.final_reward
                else:
                    for cluster in self.good_region_clusters:
                        for point, reward in cluster:
                            if (x, y) == point:
                                state_reward += reward

                    for cluster in self.bad_region_clusters:
                        for point, reward in cluster:
                            if (x, y) == point:
                                state_reward += reward

                    if state_reward == 0.0:
                        state_reward = 0.5 if np.random.random() < self.sparsity else 0.0

                state_rewards[(x, y)] = state_reward

        return state_rewards


    def step(self, action):
        x, y = self.agent_position

        # Get the reward based on the current state and policy context
        reward = self.state_rewards.get((x, y), 0)

        if action == "up" and y < self.height - 1:
            y += 1
        elif action == "down" and y > 0:
            y -= 1
        elif action == "left" and x > 0:
            x -= 1
        elif action == "right" and x < self.width - 1:
            x += 1

        # Update agent position
        self.agent_position = (x, y)

        # Get the reward based on the current state and policy context
        reward = self.state_rewards.get(self.agent_position, 0)


        if self.agent_position in self.end:
            done = True
        else:
            done = False

        # Get the reward for the updated position and policy context
        updated_reward = self.state_rewards.get(self.agent_position, 0)

        # Check if the new position is the end state
        done = (self.agent_position == self.end)

        return self.agent_position, updated_reward, done

