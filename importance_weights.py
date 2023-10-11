from policy_functions import get_quadrant_policy

def calculate_importance_weights(eval_policy, behav_policy, behavior_policies, gridworld):
    all_weights = []
    for trajectory in behavior_policies:
        cum_ratio = 1
        cumul_weights = []
        for step in trajectory:
            eval_action_probs = get_quadrant_policy(step[0], eval_policy, gridworld)
            behav_action_probs = get_quadrant_policy(step[0], behav_policy, gridworld)
            ratio = (0.8*eval_action_probs[step[1]] +0.2*0.25)/ (0.8*behav_action_probs[step[1]]+0.2*0.25)
            cum_ratio *= ratio
            cumul_weights.append(cum_ratio)
        all_weights.append(cumul_weights)

    return all_weights
