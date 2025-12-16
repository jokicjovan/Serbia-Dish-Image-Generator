import torch
from torch import cat, clamp
from torch import nn
from torch.nn import Module, Sequential, Conv2d, Dropout2d, BatchNorm2d, Flatten, Linear, ConvTranspose2d, ReLU, Tanh, \
    LeakyReLU, init


class Encoder(Module):
    """Encoder component"""
    def __init__(self, latent_dim, img_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.img_dim = img_dim
        self.encoder_input = self.img_dim[-1] + 512
        self.layers = Sequential(
            Conv2d(self.encoder_input, 64, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2),
            Dropout2d(0.1),

            Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2),
            Dropout2d(0.1),

            Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            LeakyReLU(0.2),
            Dropout2d(0.1),

            Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(512),
            LeakyReLU(0.2),
            Flatten(),
        )

        self.mean = Linear(512 * 4 * 4, latent_dim)
        self.log_of_variance = Linear(512 * 4 * 4, latent_dim)


    def forward(self, img, embedding):
        embedding = embedding.view(embedding.size(0), 512, 1, 1).repeat(1, 1, self.img_dim[0], self.img_dim[1])
        result = cat((img, embedding), dim=1)
        x = self.layers(result)
        mean = self.mean(x)
        log_of_var = clamp(self.log_of_variance(x), min=-10, max=10)
        return mean, log_of_var


class Decoder(Module):
    '''Decoder component'''
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.decoder_input = Linear(latent_dim + 512, 512 * 4 * 4)
        self.layers = Sequential(
            ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(),

            ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(),

            ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(64),
            ReLU(),

            ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            Tanh()
        )

    def forward(self, z, clip_embedding):
        z = cat([z, clip_embedding], dim=1)
        x = self.decoder_input(z)
        x = x.view(-1, 512, 4, 4)
        x = self.layers(x)
        return x


class ConvCVAE(Module):
    """Conditional Variational Autoencoder."""

    def __init__(self, img_dim=[64, 64, 3], latent_dim=128):
        super(ConvCVAE, self).__init__()
        self.img_size = img_dim[0]
        self.latent_dim = latent_dim

        self.encoder = Encoder(img_dim=img_dim,latent_dim=latent_dim)

        self.decoder = Decoder(latent_dim=latent_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (Conv2d, ConvTranspose2d)):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        z = mean + e * std
        return z

    def encode(self, x, embedding):
        return self.encoder(x, embedding)

    def decode(self, z, embedding):
        return self.decoder(z, embedding)

    def forward(self, x, embedding):
        mean, log_var = self.encode(x, embedding)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z, embedding)

        return recon_x, mean, log_var

    def generate(self, embedding, num_samples=1):
        embedding = embedding.unsqueeze(0).repeat(num_samples, 1) if len(embedding.shape) == 1 else embedding

        z = torch.randn(embedding.size(0), self.latent_dim).to(embedding.device)

        with torch.no_grad():
            generated = self.decode(z, embedding)

        return generated

    def reconstruct(self, x, clip_embedding):
        with torch.no_grad():
            mean, _ = self.encode(x, clip_embedding)
            recon = self.decode(mean, clip_embedding)
        return recon

