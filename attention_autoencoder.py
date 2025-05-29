import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple feature-wise attention module.
# Inherit the basic module nn.Module from Pytorch.
# It is a sub-module which can register parameters ancd execute forward propogation.
class FeatureAttention(nn.Module):
    
    # Define initialization, accept one parameter: input_dim - the dimentions of the input features.
    def __init__(self, input_dim):
        
        # Utilize the constructor from the father class nn.Module.
        # Complete the necessary inner registration.
        super(FeatureAttention, self).__init__()
        
        # Learnable weights for each feature.
        # Define a fully-connected layer self.att.
        # This layer contains a group of learnable weights and bias.
        # They are used to do the linear transformation for each feature in every sample data.
        self.att = nn.Linear(input_dim, input_dim)

    # Define the method of forward propogation, accept input x.
    def forward(self, x):
        
        # x: [batch_size, hidden_dim] (here, hidden_dim = input_dim)
        # First, self.att(x) is to do the linear transformation of x.
        # lin(x) = x W^T + b.
        # Then enter the output of the linear transformation into the sigmoid function, to make the output /belong to (0,1).
        # And the shape of weights still is [batch_size, hidden_dim], but each element belong from 0 to 1.
        weights = torch.sigmoid(self.att(x)) 
        
        # element-wise attention
        # Apply the weight to every feature channel.
        # Realize attention mechanism.
        return x * weights 

# Attention-based autoencoder for anomaly detection.
# This class inherites from the nn.Module from PyTorch.
# It is used to build a learnable neural network module.
class AttentionAutoencoder(nn.Module):

    # The constructor function, accept 1) input dimensions 2) the dimention of the latent space.
    def __init__(self, input_dim, hidden_dim):

        # Utilize the initialization logic of the father class.
        super(AttentionAutoencoder, self).__init__()

        # Encoder: compress input.
        # First, map the input linearly to the hidden_dim.
        # Then, use activation function - ReLU to activate the mapping output.
        self.encoder = nn.Sequential( nn.Linear(input_dim, hidden_dim), nn.ReLU() )
        
        # Attention on latent representation.
        # Initiate the previously defined FeatureAttention module.
        # In order to perform feature-wise attention weighting on the hidden_dim-dimensional latent representation.
        self.attention = FeatureAttention(hidden_dim)
        
        # Decoder: reconstruct input.
        # A single-layer linear transformation.
        # It maps the weighted latent vector back to the original input_dim dimensionality for reconstructing the input.
        self.decoder = nn.Linear(hidden_dim, input_dim)

    # Define the forward propogation interface.
    # x is the input vetor with shape [batch_size, input_dim].
    def forward(self, x):

        # x: [batch_size, input_dim]
        z = self.encoder(x)

        # Apply with the weights then get the attention.
        z_att = self.attention(z)

        # Pass the weighted latent vector through the decoder to reconstruct recon.
        # It is an ouput with the same dimensionality as x.
        recon = self.decoder(z_att)

        # The final output of the forward pass (the reconstruction error).
        # It will be used for subsequent conputations (such as the MSE loss or anomaly score).
        return recon


def train_model(model, data_loader, num_epochs=20, lr=1e-3, device='cpu'):
    '''
        funtion: Define the function which is used to train the input model.
        Parameters:
            model: the model that is going to be trained.
            data_loader: provide the training data batch by batch.
            num_epocs: the number of traning epoches.
            lr: the learing rate, by default is 0.001.
            device: the device that is used for the training, be default is CPU.
    '''

    # Copy the model and all parameters to the device.
    # The continuous forward as well as the back calculations will take place on this device.
    model.to(device)

    # Initialize an optimazer "Adam".
    # It is used to update the learnable parameters - model.parameters.
    # The learning rate is 'lr'.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss function - Mean Squared Error.
    # It is used to measure the discrepency between the model's outputrecon and the true input x.
    criterion = nn.MSELoss()

    # Train this model.
    model.train()

    # Go through every training epoch.
    for epoch in range(1, num_epochs + 1):

        # Before this epoch starts, set the total_loss as 0.
        # It is used to record the total loss of all batches in this epoch.
        total_loss = 0.0

        # for each batch
        for batch in data_loader:
            # batch: [batch_size, input_dim]
            x = batch[0].to(device)

            # Use forward method get recon.
            recon = model(x)

            # Calculate the MSE to get the loss.
            loss = criterion(recon, x)

            # Clear the previous grad, prepare for this back-propogation.
            optimizer.zero_grad()

            # Execute the back propogation, Calculate all gradients correspond to all the learnable parameters.
            loss.backward()

            # According to the calculated gradients, use Adam to update parameters, to let the loss goes down.
            optimizer.step()

            # Add the loss of this batch to the total loss.
            # x.size(0) means: to calculate the average loss easier in the following step.
            total_loss += loss.item() * x.size(0)

        # After go through all batches in this epoch, calculate the average loss.
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")


def compute_anomaly_score(model, x, device='cpu'):
    '''
        Function: Anomaly score (reconstruction error).
        Parameters:
            model: the trained auto-encoder model.
            x: input vector.
            device: by default is cpu.
        Return: the anomaly score.
    '''
    model.to(device)

    # Trun the model to evaluation mode.
    model.eval()

    # Forbid the gradients.
    # Only do the forward calculation. No backward.
    with torch.no_grad():
        x = x.to(device)
        recon = model(x)
        # Mean squared error per sample
        scores = torch.mean((recon - x) ** 2, dim=1)
    
    return scores

if __name__ == '__main__':
    # Hyperparameters
    HIDDEN_DIM = 8
    BATCH_SIZE = 32
    NUM_EPOCHS = 20

    df = pd.read_csv('data.txt', sep=r'\s+', engine='python')

    feature_cols = [
        'linear_velocity',
        'angular_velocity',
        'gps_x', 'gps_y', 'gps_yaw',
        'lidar_x', 'lidar_y', 'lidar_yaw',
        'imu_a', 'imu_w'
    ]
    features = df[feature_cols].values

    data_tensor = torch.tensor(features, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(data_tensor)
    
    # Initialize and train the model
    model = AttentionAutoencoder(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM)

    split = int(0.8 * len(data_tensor))
    train_ds = torch.utils.data.TensorDataset(data_tensor[:split])
    test_ds = torch.utils.data.TensorDataset(data_tensor[split:])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE)

    train_model(model, train_loader, num_epochs=NUM_EPOCHS)

    all_scores = []

    for (x_batch, ) in test_loader:
        scores = compute_anomaly_score(model, x_batch)
        all_scores.append(scores)
    
    all_scores = torch.cat(all_scores)
    print('Test-set anomaly scores:', all_scores)
    print("min:", all_scores.min().item())
    print("max:", all_scores.max().item())
    print("mean:", all_scores.mean().item())
    print("std:", all_scores.std().item())


# Usage outline:
# 1. Prepare a DataLoader `train_loader` with normal (non-spoofed) fused sensor data.
# 2. Instantiate and train:
#       model = AttentionAutoencoder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM)
#       train_model(model, train_loader)
# 3. For a new batch `x_test`, compute scores:
#       scores = compute_anomaly_score(model, x_test)
# 4. Declare spoofing if score > chosen_threshold
