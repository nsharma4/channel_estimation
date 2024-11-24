import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import MatDataset
from pathlib import Path

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)  # Mean of latent space
        self.fc_logvar = nn.Linear(64, latent_dim)  # Log variance of latent space

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def encode(self, x):
        """Encode the input to the latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterize to sample z from the latent space."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from the latent space to the output."""
        return self.decoder(z)

    def forward(self, x):
        """Define the forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """VAE Loss function: Reconstruction + KL Divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div

def train_vae(vae, train_loader, val_loader, epochs, lr, device):
    """Train the VAE on the given dataloader."""
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    vae.to(device)

    for epoch in range(epochs):
        vae.train()
        total_train_loss = 0
        for h_estimated, h_ideal, _ in train_loader:
            h_estimated, h_ideal = h_estimated.to(device), h_ideal.to(device)

            # Reshape data for input into VAE
            batch_size = h_estimated.size(0)
            input_data = h_estimated.view(batch_size, -1)
            target_data = h_ideal.view(batch_size, -1)

            optimizer.zero_grad()
            recon, mu, logvar = vae(input_data)
            loss = loss_function(recon, target_data, mu, logvar)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Validation
        vae.eval()
        total_val_loss = 0
        with torch.no_grad():
            for h_estimated, h_ideal, _ in val_loader:
                h_estimated, h_ideal = h_estimated.to(device), h_ideal.to(device)

                batch_size = h_estimated.size(0)
                input_data = h_estimated.view(batch_size, -1)
                target_data = h_ideal.view(batch_size, -1)

                recon, mu, logvar = vae(input_data)
                loss = loss_function(recon, target_data, mu, logvar)
                total_val_loss += loss.item()

        print(f"Epoch {epoch + 1}, Train Loss: {total_train_loss / len(train_loader)}, "
              f"Val Loss: {total_val_loss / len(val_loader)}")

def main():
    # Dataset configuration
    DATA_DIR = Path("C:/Users/neals/Desktop/CS_274E_Deep_Generative_Models/274 Group Project/channel estimation/small_dataset/small_dataset")
    TRAIN_DIR = DATA_DIR / "train"
    VAL_DIR = DATA_DIR / "val"
    PILOT_DIMS = (18, 2)
    BATCH_SIZE = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load train and validation datasets
    train_dataset = MatDataset(data_dir=TRAIN_DIR, pilot_dims=PILOT_DIMS, return_type="complex")
    val_dataset = MatDataset(data_dir=VAL_DIR, pilot_dims=PILOT_DIMS, return_type="complex")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # VAE configuration
    input_dim = 18 * 2  # Flattened pilot dimensions
    latent_dim = 32  # Latent space dimension
    output_dim = 120 * 14  # Flattened full channel dimensions
    vae = VAE(input_dim, latent_dim, output_dim)

    # Training parameters
    epochs = 20
    lr = 1e-3

    # Train the VAE
    train_vae(vae, train_loader, val_loader, epochs, lr, device)

if __name__ == "__main__":
    main()
