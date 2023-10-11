from importance_weights import calculate_importance_weights
import numpy as np
import torch

def variance_terms_tens(eval_policy, behav_policy, behavior_policies, gridworld):
  # Initialize lists to store axis data for each policy
  t = []
  s = []
  s_last = []
  a = []
  r = []
  g_last = []
  w_last = []
  w = calculate_importance_weights(eval_policy, behav_policy, behavior_policies, gridworld)
  psi = []

  for index, policy in enumerate(behavior_policies):
      policy_array = np.array(policy)
      t.append(policy_array[:, 6].astype(int))
      # s.append(policy_array[:, 0])

      # last timestep for gamma
      g_last.append(len(policy))
      # last importance weight
      w_last.append(w[index][-1])


      s.append(policy_array[:, 0][1:])
      psi.append(policy_array[:,5][1:])
      s_last.append(policy_array[:,0][-1])
      a.append(policy_array[:, 1])
      r.append(policy_array[:, 2].astype(float))


  s_w_diff = []
  for index, weight in enumerate(w):
    # diff = np.array(w[index][:-1]) - np.array(w[index][1:])
    diff = np.array(weight[:-1]) - np.array(weight[1:])
    s_w_diff.append(diff)

  gtrw = np.power(0.9,t)*r*np.array(w)
  gw_l = np.power(0.9, g_last)*w_last
  # Number of bootstrap iterations
  num_iterations = 1000

  np.random.seed(0)
  # Get bootstrap indices
  bootstrap_indices = np.random.choice(len(behavior_policies), size=(num_iterations, len(behavior_policies)), replace=True)

  gtrw = np.power(0.9,t)*r*np.array(w)
  samples_IS = np.take(gtrw, bootstrap_indices, axis = 0)
  samples_s = np.take(s, bootstrap_indices, axis = 0)
  samples_w_diff = np.take(s_w_diff, bootstrap_indices, axis = 0)
  samples_last = np.take(gw_l, bootstrap_indices, axis = 0)

  IS_all = np.array([np.sum(np.concatenate(arr), axis=0)/len(behavior_policies) for arr in samples_IS])
  F_all = np.array([np.sum((arr), axis=0)/len(behavior_policies) for arr in samples_last])

  ft = torch.tensor(F_all).reshape(-1,1)
  f_res = [tensor for tensor in ft]
  last_first_tens_arr = np.array(f_res)

  It = torch.tensor(IS_all).reshape(-1,1)
  IS_res = [tensor for tensor in It]
  IS_tens_arr = np.array(IS_res)

  ss = [np.concatenate(p_set) for p_set in samples_s]
  state_tensors = [
    [torch.tensor(sample, dtype=torch.float32) for sample in sublist]
    for sublist in ss]

  # sample_weights = [torch.tensor(np.concatenate(p_set)).reshape(-1,1) for p_set in samples_w_diff]

  # Initialize an empty array to store the tensors
  w_diff_tensors = np.empty((len(samples_w_diff),), dtype=object)

  for i, p_set in enumerate(samples_w_diff):
      # Concatenate the NumPy arrays and reshape the resulting tensor
      ssw_tensor = torch.tensor(np.concatenate(p_set)).reshape(-1, 1)

      # Store the tensor directly into w_diff_tensors
      w_diff_tensors[i] = ssw_tensor


  state_tensor_og = [[torch.tensor(state_t, dtype = torch.float32) for state_t in traj_t] for traj_t in s]

  psi_og = [[torch.tensor(state, dtype = torch.float32) for state in traj] for traj in psi]
  psi_arrays = np.empty((len(psi_og),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_psi = [torch.stack(sublist, dim=0) for sublist in psi_og]

  for i, tensor in enumerate(stacked_psi):
      # psi = net(tensor)  # Process each tensor separately
      # Store the psi tensor directly into psi_arrays
      psi_arrays[i] = tensor
  reshaped_psi = [tensor.unsqueeze(1) for tensor in psi_arrays]


  return IS_tens_arr, state_tensors , w_diff_tensors, last_first_tens_arr, state_tensor_og, reshaped_psi