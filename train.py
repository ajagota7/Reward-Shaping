from loss_functions import calc_variance_loss, mse
import torch.optim as optim


def train_mse_var(net, num_epochs, learning_rate, mse_ratio, var_ratio, IS, state_tensors, w_diff, f, phi_set, st_og, psi_res):
  net.train()
  # Define the optimizer (Adam optimizer)
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)

  # Training loop
  for epoch in range(num_epochs):
    total_loss = 0
    variance_loss = calc_variance_loss(IS, state_tensors, w_diff, f, net, phi_set)
    mse_loss = mse(net, st_og, psi_res)/len(st_og)
    print(f"Epoch {epoch+1}")
    print("Var loss: ", variance_loss)
    print("Feature loss (MSE): ", mse_loss)
    tot = mse_ratio*mse_loss + var_ratio*variance_loss
    # tot = variance_loss
    # tot = mse_loss


    # Backpropagation and optimization for the trajectory
    optimizer.zero_grad()
    tot.backward()
    optimizer.step()

    # print("Total Loss: ", tot)

    total_loss += tot.item()

    print(f"Total Loss: {total_loss}")
    print("-" * 40)

  # Print the weights of the neural network
  for name, param in net.named_parameters():
    if param.requires_grad:
        print(f"Parameter name: {name}")
        print(f"Weights: {param.data}")
  # results = SCOPEnet(scope_set, 30000, eval_policy, behav_policy, net)
  # print("scope_results: ", results)

  return net
