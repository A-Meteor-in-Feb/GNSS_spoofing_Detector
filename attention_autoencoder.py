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

# Attention-based autoencoder for anomaly detection
class AttentionAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionAutoencoder, self).__init__()
        # Encoder: compress input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Attention on latent representation
        self.attention = FeatureAttention(hidden_dim)
        # Decoder: reconstruct input
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [batch_size, input_dim]
        z = self.encoder(x)
        z_att = self.attention(z)
        recon = self.decoder(z_att)
        return recon

# Example training loop stub
def train_model(model, data_loader, num_epochs=20, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch in data_loader:
            # batch: [batch_size, input_dim]
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}")

# Anomaly score (reconstruction error)
def compute_anomaly_score(model, x, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        recon = model(x)
        # Mean squared error per sample
        scores = torch.mean((recon - x) ** 2, dim=1)
    return scores

if __name__ == '__main__':
    # Hyperparameters
    FEATURE_DIM = 16
    HIDDEN_DIM = 8
    NUM_SAMPLES = 1000
    BATCH_SIZE = 32
    NUM_EPOCHS = 5

    # Generate dummy normal data
    data = torch.randn(NUM_SAMPLES, FEATURE_DIM)
    dataset = torch.utils.data.TensorDataset(data)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize and train the model
    model = AttentionAutoencoder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM)
    train_model(model, loader, num_epochs=NUM_EPOCHS)

    # Test anomaly scores on new samples
    test_data = torch.randn(10, FEATURE_DIM)
    scores = compute_anomaly_score(model, test_data)
    print('Anomaly scores:', scores)

# Usage outline:
# 1. Prepare a DataLoader `train_loader` with normal (non-spoofed) fused sensor data.
# 2. Instantiate and train:
#       model = AttentionAutoencoder(input_dim=FEATURE_DIM, hidden_dim=HIDDEN_DIM)
#       train_model(model, train_loader)
# 3. For a new batch `x_test`, compute scores:
#       scores = compute_anomaly_score(model, x_test)
# 4. Declare spoofing if score > chosen_threshold
