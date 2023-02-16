import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from scipy.stats import genextreme

# define Daubechies 2 wavelet filter
wavelet = 'db2'
wavelet_filter = pywt.Wavelet(wavelet)

class DataAugmentation(nn.Module):
    def __init__(self, sigma=0.1):
        super(DataAugmentation, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        # apply direct wavelet transform to get wavelet and approximation coefficients
        cA, cD = pywt.dwt(x, wavelet_filter)

        # perturb the wavelet (detail) coefficients by adding small random noise
        cD_perturbed = cD + torch.randn(cD.shape) * self.sigma

        # apply inverse wavelet transform to get back the signal in the time domain
        x_perturbed = pywt.idwt(cA, cD_perturbed, wavelet_filter)

        # return perturbed signal
        return x_perturbed


class Autoencoder(nn.Module):
    def __init__(self, input_size=100, hidden_size=10, latent_size=2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ContrastiveLearning(nn.Module):
    def __init__(self, input_size=2, hidden_size=10, output_size=2):
        super(ContrastiveLearning, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        # encode input
        encoded = self.encoder(x)
        # project encoded input to a lower-dimensional space
        projected = self.projection_head(encoded)
        # normalize projected values for contrastive loss
        normalized = F.normalize(projected, dim=1)
        return normalized


# Define the loss functions
class LossFunctions:
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # Reconstruction loss
    def reconstruction_loss(self, x, x_hat):
        return torch.mean((x - x_hat)**2)

    # Contrastive loss
    def contrastive_loss(self, z_i, z_j, temperature=0.5):
        z = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.exp(torch.mm(z, z.t()) / temperature)
        mask = torch.eye(len(z)*2, len(z)*2)
        numerator = mask * similarity_matrix
        denominator = mask * similarity_matrix.sum(dim=1, keepdim=True)
        loss_per_sample = -torch.log(numerator / denominator + 1e-8)
        loss = loss_per_sample.sum() / (len(z)*(len(z)-1))
        return loss

    # Tail-weighted distance loss
    def tail_weighted_distance_loss(self, x, x_hat):
        ecdf = torch.linspace(0, 1, len(x))
        ecdf_hat = torch.linspace(0, 1, len(x_hat))
        twd = torch.sum(torch.abs(ecdf - ecdf_hat) * (1 - ecdf) ** self.alpha)
        return twd

    # Cramer-von Mises distance loss
    def cramer_von_mises_distance_loss(self, x, x_hat):
        def compute_gpd_params(data):
            # Fit the Generalized Pareto Distribution (GPD) to the block maxima
            # of the original data
            threshold = np.percentile(data, 90)
            block_maxima = data[data >= threshold]
            params = genpareto.fit(block_maxima)
            return params

        def compute_gev_params(data):
            # Fit the Generalized Extreme Value (GEV) Distribution to the block maxima
            # of the original data
            threshold = np.percentile(data, 90)
            block_maxima = data[data >= threshold]
            params = genextreme.fit(block_maxima)
            return params

# Define the projection head
class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, output_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
 
class CombinedLoss(nn.Module):
    def __init__(self, lambda1, lambda2, lambda3):
        super(CombinedLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, loss_contrastive, loss_reconstruction, loss_distribution):
        return self.lambda1 * loss_contrastive + self.lambda2 * loss_reconstruction + self.lambda3 * loss_distribution
   