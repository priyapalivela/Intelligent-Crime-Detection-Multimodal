import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    """
    Audio feature extractor using CNN + Bidirectional LSTM on MFCC features.
    
    Input shape: [batch, 40, time] or [batch, time, 40] (will auto-transpose if needed)
    Output shape: [batch, 256]  (bidirectional LSTM with hidden_dim=128 → 256 features)
    """
    def __init__(self, input_dim=40, hidden_dim=128):
        super(AudioEncoder, self).__init__()
        
        # CNN part - feature extraction from MFCC
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.out_dim = hidden_dim * 2  # 256 if hidden_dim=128

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: torch.Tensor [batch, channels=40, time] or [batch, time, channels]
            
        Returns:
            torch.Tensor [batch, 256]
        """
        # Safety: make sure channels are second dimension [B, C, T]
        if x.dim() == 3 and x.shape[1] != 40:
            print("Warning: Transposing MFCC input to [B, channels=40, time]")
            x = x.transpose(1, 2)

        # CNN blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)

        # Prepare for LSTM: [B, T', 128] → [B, T', C]
        x = x.transpose(1, 2)

        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Global average pooling over time
        features = lstm_out.mean(dim=1)
        features = self.layer_norm(features)

        return features


if __name__ == "__main__":
    # Quick test
    encoder = AudioEncoder()
    dummy_input = torch.randn(2, 40, 100)   # normal case
    dummy_input_wrong = torch.randn(2, 100, 40)  # transposed case
    
    out1 = encoder(dummy_input)
    out2 = encoder(dummy_input_wrong)
    
    print("Output shape (normal):", out1.shape)     # should be [2, 256]
    print("Output shape (transposed):", out2.shape)  # should also be [2, 256]


