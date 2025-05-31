### Encoder

Encoder = Fully-connected layer + ReLU activation function

The learnable paramter $W_e \in \mathbb{R}^{H \times D}$, $b_e \in \mthbb{R}^H$. H means the hidden_dim.

For the whole batch of input $X \in \mathbb{R}^{B\timesD}$, after the input data go though the hidden layer, we can get $Z = ReLU(XW_e^T + 1_Bb_e^T) \in \mathbb{R}^{B\timesH}$, where $1_B \in \mathbb{R}^B$ means broadcats the bias $b_e$ to each line.

### Feature Attention

The function of the feature attention is to assign a "weight" ($W_a\in \mathbb{R}^{H\timesH},\ b_a \in \mathbb{R}^H$) which $\in$ (0,1) to each weight. This can help us to know which features are more important.

For the whole batch Z, after they go through this feature attention function, we can get: $W = \sigma(ZW_a^T+1_Bb_a^T)\in \mathbb{R}^{B\timesH}$.
（Here, the encoder's output hidden representation Z is used as the input to the attention network to dynamically compute the weights W for each sample and each hidden dimension, thereby "examining the features of the current sample" to "decide which hidden channels to focus on."）

Then we do $Z_{att} = Z \bigodot W$.
(By element-wise multiplying the weights W computed in the previous step with the original hidden representation Z, the 'unimportant' or 'noisy' hidden dimensions are directly suppressend while the 'key' hidden dimensions are aplified or retrained, enabling the decoder to focus solely on reconstructing the most representative features.)

### Decoder

A single layer linear mapping: $\hat{x} = Z_{att} W_d^T + 1_Bb_d^T \in \mathbb{R}^{B\timesD}$.

### Loss Function

We use MSE here, because we have the input data and the reconstructed data.

### Learning rate

We use the Adam method to dynamically adjust it.