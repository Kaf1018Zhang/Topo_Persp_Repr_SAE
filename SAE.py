import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon, latent

def sparse_loss(latent, l1_weight=1e-4):
    return l1_weight * torch.mean(torch.abs(latent))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_sae(model, data_tensor, epochs=20, batch_size=256, lr=8e-2, l1_weight=1e-4):
    from torch.utils.data import DataLoader, TensorDataset

    dataset = DataLoader(TensorDataset(data_tensor), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in dataset:
            batch = batch.to(device)
            recon, latent = model(batch)
            loss_recon = loss_fn(recon, batch)
            loss_sparse = sparse_loss(latent, l1_weight)
            loss = loss_recon + loss_sparse
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    
    return model
