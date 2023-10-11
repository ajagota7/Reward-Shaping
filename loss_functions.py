import torch
import numpy as np 
import torch
import torch.nn as nn


def calc_variance_loss(IS, state_tensors, w_diff, f, feature_net, behavior_policies):

  output_arrays = np.empty((len(state_tensors),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_samples = [torch.stack(sublist, dim=0) for sublist in state_tensors]

  for i, tensor in enumerate(stacked_samples):
      output = feature_net(tensor)  # Process each tensor separately

      # Store the output tensor directly into output_arrays
      output_arrays[i] = output

  phi_w_diff_arrays = output_arrays*w_diff
  # Assuming phi_w_diff_arrays is a list of PyTorch tensors
  phi_diff_sums_array = np.empty(len(phi_w_diff_arrays), dtype=object)

  for i, p_set in enumerate(phi_w_diff_arrays):
      phi_diff_sum = torch.sum(p_set, dim=0) / len(behavior_policies)
      phi_diff_sums_array[i] = phi_diff_sum

  state_0 = feature_net(torch.tensor((0,0), dtype = torch.float32))
  state_end = feature_net(torch.tensor((4,4), dtype = torch.float32))
  f_all = phi_diff_sums_array + (state_end.item()*np.array(f) - state_0.item())
  IS_phi_terms = IS*f_all
  IS_sq = torch.mean(torch.stack(list(IS**2)), dim = 0)
  IS_sq_all = torch.mean(torch.stack(list(IS)), dim = 0)**2
  IS_phi_l_f = torch.mean(torch.stack(list(IS_phi_terms)), dim = 0)
  IS_and_phi_l_f = torch.mean(torch.stack(list(IS)), dim = 0)*torch.mean(torch.stack(list(f_all)), dim = 0)
  phi_w_sq = torch.mean(torch.stack(list(phi_diff_sums_array**2)), dim = 0)
  phi_w_sq_all = torch.mean(torch.stack(list(phi_diff_sums_array)), dim = 0)**2

  var_IS = IS_sq - IS_sq_all
  var_scope = IS_sq + 2*IS_phi_l_f + phi_w_sq - IS_sq_all - 2*IS_and_phi_l_f - phi_w_sq_all

  # return mse_loss, var_scope
  return var_scope
  # return var_IS, var_scope


def mse(feature_net, state_tensor_og, psi_reshaped):

  state_og_arrays = np.empty((len(state_tensor_og),), dtype=object)
  # Stack the tensors in each sublist along a new dimension (assuming each sublist has the same number of tensors)
  stacked_og = [torch.stack(sublist, dim=0) for sublist in state_tensor_og]

  for i, tensor in enumerate(stacked_og):
      output_og = feature_net(tensor)  # Process each tensor separately

      # Store the output tensor directly into state_og_arrays
      state_og_arrays[i] = output_og
  # Initialize an empty array to store the outputs

  # Calculate the mean squared error (MSE) loss
  mse_loss = nn.MSELoss()

  # Calculate the loss for each pair of tensors and then take the mean
  loss = torch.mean(torch.stack([mse_loss(output, target) for output, target in zip(state_og_arrays, psi_reshaped)]))

  mse_loss = loss #/len(np.concatenate(psi_reshaped))#len(behavior_policies)

  return mse_loss

