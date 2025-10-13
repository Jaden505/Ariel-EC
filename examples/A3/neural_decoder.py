from torch import nn
import torch

class ControllerDecoder(nn.Module):
    def __init__(self, latent_dim=8, hidden_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim + 2 + 1, hidden_dim),  # latent[i] + (i, j) + layer_id
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # output: single weight value
        )

    def forward(self, latent, input_size, output_size):
        def generate_weights(in_dim, out_dim, layer_id):
            weights = []
            for i in range(in_dim):
                for j in range(out_dim):
                    emb_i = latent[i]  # use ith row
                    pos = torch.tensor([i / in_dim, j / out_dim])
                    layer = torch.tensor([layer_id])
                    input_vec = torch.cat([emb_i, pos, layer])
                    w_ij = self.fc(input_vec)
                    weights.append(w_ij)
            return torch.stack(weights).view(in_dim, out_dim)

        w1 = generate_weights(input_size, 8, layer_id=1)
        w2 = generate_weights(8, 8, layer_id=2)
        w3 = generate_weights(8, output_size, layer_id=3)

        return torch.cat([w1.flatten(), w2.flatten(), w3.flatten()])
