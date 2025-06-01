import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# LSTM based Autoencoder
class LSTMAutoencoder(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        '''
            Func: LSTM-based autoencoder.
            Param:
                iuput_dim: Feature dimension - D.
                hidden_dim: Hidden dimension - H.
                num_layers: the layer of LSTM, by default is 1.
        '''
        super(LSTMAutoencoder, self).__init__()

        # encoder
        self.encoder_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)
        
        # feature attention
        self.attention = FeatureAttention(hidden_dim)

        # decoder
        self.decoder_lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=False)

        '''
            one question here, what's the functions of 'first_batch & bidirection' ???
        '''

        # linear layer to map the output from the decoder to D dimension
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            Func: Forward calculation
            Param:
                x: [B, T, D]
            Return: recon: [B, T, D]
        '''
        batch_size, seq_len, last_hidden_state = x.size()

        '''
            one question here, what's the meaning of s_???
        '''
        
        # encoder_output:[B, T, H]
        # (hidden_n, cell_n): the hidden state and the cell state of the last layer
        encoder_output, (hidden_n, cell_n) = self.encoder_lstm(x)

        # hidden state of the autoencoder
        z = hidden_n[-1]
        

        # use attention -- for now, no use.
        z_att = self.attention(z)

        # decoder - input, repeat z_att T times (?) -> a sequence
        decoder_input = z_att.unsqueeze(1).repeat(1, seq_len, 1)
        # decoder - get the decoder output, actually, last_hidden_state is useless
        decoder_output, last_hidden_state = self.decoder_lstm(decoder_input)

        # output mapping
        recon = self.output_layer(decoder_output)

        return recon
    
# calculate feature attention
class FeatureAttention(nn.Module):
    
    # Define initialization, accept one parameter: input_dim - the dimentions of the input features.
    def __init__(self, input_dim):
        
        super(FeatureAttention, self).__init__()
        
        self.att = nn.Linear(input_dim, input_dim)

    # Define the method of forward propogation, accept input x.
    def forward(self, x):
        
        weights = torch.sigmoid(self.att(x)) 
        
        return x * weights 

# train model
def train_model_lstm(model: nn.Module, train_loader: DataLoader, num_epochs: int = 20, lr: float = 1e-3, device: str = "cpu"):
    '''
        Func: Train LSTM autoencoder
        Param:
            model: LSTMAutoencoder instance
            train_loader: DataLoader, which provides the training sequence [B, T, D]
            num_spochs: the number of epochs
            lr: learning rate
            device: by default 'cpu'
    '''   
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() 

    model.train()
    for epoch in range(1, num_epochs+1):
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.6f}")


def compute_anomaly_scores_lstm(model: nn.Module, data_loader: DataLoader, device: str = 'cpu') -> torch.Tensor:
    
    model.to(device)
    model.eval()
    all_scores = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)       # [B, T, D]
            recon = model(x)              # [B, T, D]
            # 计算每个序列的 MSE：先算 (recon-x)^2，得到 [B, T, D]，再对 T,D 两个维度求平均 → [B]
            scores = torch.mean((recon - x) ** 2, dim=(1, 2))
            all_scores.append(scores.cpu())

    # 拼接成一个 shape 为 [N_total] 的 1D Tensor
    return torch.cat(all_scores, dim=0)  # torch.Tensor, shape [N_test]



# Data Process
def create_sequences(df: pd.DataFrame, feature_cols: list, window_size: int, stride: int = 1) -> torch.Tensor:
    '''
        Func: Cut the specific features into sliding window sequence.
        Param:
            df: pandas.DataFrame, the table that contains data from multi-sensors.
            feature_cols: list, the columns (e.g. ['gps_x', 'gps_y', ...]).
            window_size: int, the time duration of each sequence, T.
            stride: int, the step that the sliding window slides on the data.
        Return:
            sequences: torch.FloatTensor. shape = [N, T, D]
                       where, N = (len(df) - window_size) // stride + 1
                              D = len(feature_cols)
    '''
    data = df[feature_cols].values.astype(np.float32)
    num_samples, D = data.shape

    N = (num_samples - window_size) // stride + 1

    sequences = []

    for i in range(0, N*stride, stride):
        seq = data[i: i + window_size] #(T, D)
        sequences.append(seq)

    sequences = np.stack(sequences, axis=0) #(N, T, D)
    return torch.from_numpy(sequences)




if __name__ == '__main__':
    df = pd.read_csv('data.txt', sep=r'\s+', engine='python')
    feature_cols = [
        'linear_velocity',
        'angular_velocity',
        'gps_x', 'gps_y', 'gps_yaw',
        'lidar_x', 'lidar_y', 'lidar_yaw',
        'imu_a', 'imu_w'
    ]
    
    WINDOW_SIZE = 20
    all_sequences = create_sequences(df, feature_cols, window_size=WINDOW_SIZE, stride=1)
    print("All sequences shape:", all_sequences.shape) # [25108, 20, 10]

    N, T, D = all_sequences.size()

    train_ratio = 0.8
    split = int(train_ratio * N)

    train_sequences = all_sequences[:split]
    test_sequences = all_sequences[split:]

    train_ds = TensorDataset(train_sequences)
    test_ds = TensorDataset(test_sequences)

    BATCH_SIZE = 32

    # shuffle = True: make the order randomly before each epoch
    # drop_last = True: if split is not the integer times over the BATCH_SIZE, drop the last ones
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Training sequences: {len(train_ds)}, Testing sequences: {len(test_ds)}")

    HIDDEN_DIM = 16    
    NUM_LAYERS = 1     
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-3

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Use device:", device)

    model = LSTMAutoencoder(input_dim=D, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)

    print("Start training LSTM Autoencoder ...")
    train_model_lstm(model, train_loader, num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=device)

    
    print("Computing anomaly scores on train set ...")
    train_scores = compute_anomaly_scores_lstm(model, train_loader, device=device)

    print("Computing anomaly scores on test set ...")
    test_scores = compute_anomaly_scores_lstm(model, test_loader, device=device)

    mu = train_scores.mean().item()
    sigma = train_scores.std().item()
    k = 3.0
    tau = mu + k * sigma
    print(f"Threshold (mean + {k}*std): {tau:.6f}")

    print("First 10 test scores:", test_scores[:10])

    anomalies = test_scores > tau
    print(f"Number of anomalies in test set: {anomalies.sum().item()} / {len(test_scores)}")


