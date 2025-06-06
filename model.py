import torch
import torch.nn as nn
import math

class GeneExpressionTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 1000,  # gene_max_cnt
        d_model: int = 64,     # transformer dimension
        nhead: int = 4,         # number of attention heads
        num_layers: int = 4,    # number of transformer layers
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_classes: int = 3,  # number of MOA classes
        max_seq_length: int = 1000  # maximum sequence length (gene_max_cnt)
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(1, d_model)  
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        # Add channel dimension and project to d_model
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask for padding (if needed)

        mask = None
        
        # Pass through transformer
        x = self.transformer_encoder(x, mask)  # (batch_size, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, d_model)
        
        # Project to output classes
        logits = self.output_proj(x)  # (batch_size, num_classes)
        
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

def create_model(
    input_dim: int = 1000,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 4,
    num_classes: int = 15,
    dropout: float = 0.1
) -> GeneExpressionTransformer:
    """
    Factory function to create a GeneExpressionTransformer model with default parameters.
    """
    return GeneExpressionTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        max_seq_length=input_dim,
    )

if __name__ == "__main__":
    
    batch_size = 32
    seq_len = 1000  # gene_max_cnt
    
    model = create_model()
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len)
    
    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
