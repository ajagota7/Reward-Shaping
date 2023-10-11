from importance_weights import calculate_importance_weights
import numpy as np 
import torch

def var_terms_is(eval_policy, behav_policy, behavior_policies, num_bootstraps):
  # Initialize lists to store axis data for each policy
  t = []
  s = []
  s_all = []
  s_next = []
  s_last = []
  a = []
  r = []
  g_last = []
  w_last = []
  w = calculate_importance_weights(eval_policy, behav_policy, behavior_policies)
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
      s_all.append(policy_array[:, 0])
      s_all.append(policy_array[:, 3])
      psi.append(policy_array[:,5][1:])
      s_last.append(policy_array[:,0][-1])
      a.append(policy_array[:, 1])
      r.append(policy_array[:, 2].astype(float))

  gtrw = np.power(0.9,t)*r*np.array(w)
  gw_l = np.power(0.9, g_last)*w_last
  # Number of bootstrap iterations
  num_bootraps = num_bootstraps

  np.random.seed(0)
  # Get bootstrap indices
  bootstrap_indices = np.random.choice(len(behavior_policies), size=(num_bootraps, len(behavior_policies)), replace=True)

  gtrw = np.power(0.9,t)*r*np.array(w)
  samples_IS = np.take(gtrw, bootstrap_indices, axis = 0)
  # samples_s = np.take(s, bootstrap_indices, axis = 0)
  # samples_w_diff = np.take(s_w_diff, bootstrap_indices, axis = 0)
  # samples_last = np.take(gw_l, bootstrap_indices, axis = 0)

  IS_all = np.array([np.sum(np.concatenate(arr), axis=0)/len(behavior_policies) for arr in samples_IS])
  # F_all = np.array([np.sum((arr), axis=0)/len(behavior_policies) for arr in samples_last])

  # ft = torch.tensor(F_all).reshape(-1,1)
  # f_res = [tensor for tensor in ft]
  # last_first_tens_arr = np.array(f_res)

  It = torch.tensor(IS_all).reshape(-1,1)
  IS_res = [tensor for tensor in It]
  IS_tens_arr = np.array(IS_res)

  IS_var_all = torch.var(torch.stack(list(IS_tens_arr)), dim = 0)
  IS_mean_all = torch.mean(torch.stack(list(IS_tens_arr)), dim = 0)
  return IS_var_all.item(), IS_mean_all.item()




def var_terms_scope(eval_policy, behav_policy, behavior_policies, feature_net, num_bootraps):
  # Initialize lists to store axis data for each policy
  t = []
  s = []
  s_all = []
  s_next = []
  s_last = []
  a = []
  r = []
  g_last = []
  w_last = []
  w = calculate_importance_weights(eval_policy, behav_policy, behavior_policies)
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
      s_all.append(policy_array[:, 0])
      s_next.append(policy_array[:, 3])
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
  # gw_l = np.power(0.9, g_last)*w_last
  # Number of bootstrap iterations
  num_bootraps = 30000
  np.random.seed(0)
  # Get bootstrap indices
  bootstrap_indices = np.random.choice(len(behavior_policies), size=(num_bootraps, len(behavior_policies)), replace=True)


  state_tensor_next_og = [[torch.tensor(state, dtype = torch.float32) for state in traj] for traj in s_next]
  state_og_next_arrays = np.empty((len(state_tensor_next_og),), dtype=object)

  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_samples_next_og = [torch.stack(sublist, dim=0) for sublist in state_tensor_next_og]
  for i, tensor_i in enumerate(stacked_samples_next_og):
      output = feature_net(tensor_i).detach().numpy()  # Process each tensor separately

      # Store the output tensor directly into state_og_next_arrays
      state_og_next_arrays[i] = output

  state_tensor_og = [[torch.tensor(state, dtype = torch.float32) for state in traj] for traj in s_all]
  state_og_arrays = np.empty((len(state_tensor_og),), dtype=object)

  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_samples_og = [torch.stack(sublist, dim=0) for sublist in state_tensor_og]
  for j, tensor_j in enumerate(stacked_samples_og):
      output = feature_net(tensor_j).detach().numpy()  # Process each tensor separately

      # Store the output tensor directly into state_og_arrays
      state_og_arrays[j] = output


  state_0 = feature_net(torch.tensor((0,0), dtype = torch.float32)).item()

  r_res = [r[i].reshape(-1,1) for i in range(len(r))]
  # r_res[0].shape
  sums = r_res + 0.9*state_og_next_arrays - state_og_arrays
  gtw = np.power(0.9,t)*np.array(w)
  gtw_res = [gtw[i].reshape(-1,1) for i in range(len(gtw))]
  gtw_sums = gtw_res*sums
  gtw_sums_all = [np.sum(gtw_sums[i]) for i in range(len(gtw_sums)) ]
  # np.var(gtw_sums_all)
  scope_samples = np.take(gtw_sums_all, bootstrap_indices, axis = 0)
  all_sums = np.sum(scope_samples, axis = 1)/len(behavior_policies) - state_0
  var_scope = np.var(all_sums)
  mean_scope = np.mean(all_sums)


  return var_scope, mean_scope

