import numpy as np


def random_policy():
    # Choose a random action
    return np.random.choice(["up", "down", "left", "right"])

def run_policy(agent_position, quadrant_policy_map):
    action_probs = get_quadrant_policy(agent_position, quadrant_policy_map)
    return np.random.choice(list(action_probs.keys()), p=list(action_probs.values()))

def manhattan_distance(pos1, pos2):
    # Compute the Manhattan distance between two positions
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        

def get_quadrant_policy(position, policy_map, grid_world):
    x, y = position
    mid_x = grid_world.width // 2
    mid_y = grid_world.height // 2

    if x < mid_x:
        if y < mid_y:
            return policy_map["bottom_left"]
        else:
            return policy_map["top_left"]
    else:
        if y < mid_y:
            return policy_map["bottom_right"]
        else:
            return policy_map["top_right"]
