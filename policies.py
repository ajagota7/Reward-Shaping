# Define probability distributions for each quadrant
behav_policy = {
    "top_left": {"up": 0.2, "down": 0.2, "left": 0.2, "right": 0.4},
    "top_right": {"up": 0.3, "down": 0.2, "left": 0.2, "right": 0.3},
    "bottom_left": {"up": 0.3, "down": 0.2, "left": 0.2, "right": 0.3},
    "bottom_right": {"up": 0.4, "down": 0.2, "left": 0.2, "right": 0.2}
}

# Define probability distributions for each quadrant
eval_policy = {
    "top_left": {"up": 0.1, "down": 0.1, "left": 0.1, "right": 0.7},
    "top_right": {"up": 0.4, "down": 0.1, "left": 0.1, "right": 0.4},
    "bottom_left": {"up": 0.3, "down": 0.2, "left": 0.2, "right": 0.3},
    "bottom_right": {"up": 0.55, "down": 0.15, "left": 0.15, "right": 0.15}
}
