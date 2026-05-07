"""
VAE and Multimodal VAE implementations for metabolic state learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class ConvEncoder1D(nn.Module):
    """1D Convolutional encoder for time series data."""
    
    def __init__(self, 
                 input_length: int,
                 channels: list,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 latent_dim: int = 8):
        super().__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim
        
        layers = []
        in_channels = 1
        current_length = input_length
        
        # Build convolutional layers
        for out_channels in channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                   stride=stride, padding=padding))
            layers.append(nn.ReLU())
            current_length = (current_length - kernel_size + 2 * padding) // stride + 1
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.conv_output_size = in_channels * current_length
        
        # Fully connected layers to latent distribution
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, 1, input_length) or (batch_size, input_length)
        
        Returns:
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class ConvDecoder1D(nn.Module):
    """1D Convolutional decoder for time series data."""
    
    def __init__(self,
                 output_length: int,
                 channels: list,
                 latent_dim: int = 8,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1):
        super().__init__()
        self.output_length = output_length
        self.latent_dim = latent_dim
        self.channels = channels
        
        # Calculate expected flattened size for reconstruction
        # This is a simplified version; in practice, you'd want to match the encoder
        self.conv_output_channels = channels[0]
        self.conv_output_length = output_length // (2 ** len(channels))
        if self.conv_output_length == 0:
            self.conv_output_length = output_length
        self.fc_output_size = self.conv_output_channels * self.conv_output_length
        
        # FC layer from latent to conv input
        self.fc = nn.Linear(latent_dim, self.fc_output_size)
        
        # Deconvolutional layers
        layers = []
        in_channels = channels[0]
        for out_channels in channels[1:] + [1]:
            layers.append(nn.ConvTranspose1d(in_channels, out_channels, kernel_size,
                                            stride=stride, padding=padding))
            if out_channels != 1:
                layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.deconv_layers = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch_size, latent_dim)
        
        Returns:
            reconstructed: (batch_size, output_length)
        """
        x = self.fc(z)
        x = x.view(x.size(0), self.conv_output_channels, self.conv_output_length)
        
        x = self.deconv_layers(x)
        x = x.squeeze(1)  # Remove channel dimension
        
        # Ensure correct output length
        if x.size(1) != self.output_length:
            if x.size(1) > self.output_length:
                x = x[:, :self.output_length]
            else:
                pad_size = self.output_length - x.size(1)
                x = F.pad(x, (0, pad_size))
        
        return x


class BaselineVAE(nn.Module):
    """Standard VAE on CGM data only."""
    
    def __init__(self, 
                 cgm_length: int = 288,
                 encoder_channels: list = None,
                 decoder_channels: list = None,
                 latent_dim: int = 8):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [16, 32, 64]
        if decoder_channels is None:
            decoder_channels = [64, 32, 16]
        
        self.encoder = ConvEncoder1D(cgm_length, encoder_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder1D(cgm_length, decoder_channels, latent_dim=latent_dim)
        self.latent_dim = latent_dim
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + eps * sqrt(var)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, cgm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            cgm: (batch_size, 288)
        
        Returns:
            reconstruction: (batch_size, 288)
            mu: (batch_size, latent_dim)
            logvar: (batch_size, latent_dim)
        """
        mu, logvar = self.encoder(cgm)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        
        return reconstruction, mu, logvar
    
    def encode(self, cgm: torch.Tensor) -> torch.Tensor:
        """Get latent representation."""
        mu, logvar = self.encoder(cgm)
        return self.reparameterize(mu, logvar)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from latent representation."""
        return self.decoder(z)


class MultimodalVAEWithPoE(nn.Module):
    """
    Multimodal VAE with Product-of-Experts (PoE) fusion.
    
    Combines CGM, insulin, and physiology data through separate encoders,
    then fuses their latent distributions using Product-of-Experts.
    Handles missing modalities gracefully by dropping them from the product.
    """
    
    def __init__(self,
                 cgm_length: int = 288,
                 insulin_length: int = 288,
                 physiology_dim: int = 5,
                 encoder_channels: list = None,
                 decoder_channels: list = None,
                 latent_dim: int = 8):
        super().__init__()
        
        if encoder_channels is None:
            encoder_channels = [16, 32, 64]
        if decoder_channels is None:
            decoder_channels = [64, 32, 16]
        
        self.latent_dim = latent_dim
        
        # Modality encoders
        self.cgm_encoder = ConvEncoder1D(cgm_length, encoder_channels, latent_dim=latent_dim)
        self.insulin_encoder = ConvEncoder1D(insulin_length, encoder_channels, latent_dim=latent_dim)
        
        # Physiology encoder (simple MLP)
        self.physiology_encoder = nn.Sequential(
            nn.Linear(physiology_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.physiology_mu = nn.Linear(32, latent_dim)
        self.physiology_logvar = nn.Linear(32, latent_dim)
        
        # Shared decoder for all modalities
        self.cgm_decoder = ConvDecoder1D(cgm_length, decoder_channels, latent_dim=latent_dim)
        self.insulin_decoder = ConvDecoder1D(insulin_length, decoder_channels, latent_dim=latent_dim)
        
        # Physiology decoder (MLP)
        self.physiology_decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, physiology_dim),
        )
    
    def product_of_experts(self,
                          mu_list: list,
                          logvar_list: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Product-of-Experts (PoE) fusion of Gaussian distributions.
        
        For independent Gaussians N(μᵢ, σᵢ²), the PoE is:
        μ_PoE = (Σᵢ μᵢ/σᵢ²) / (Σᵢ 1/σᵢ²)
        σ²_PoE = 1 / (Σᵢ 1/σᵢ²)
        """
        if len(mu_list) == 0:
            # Return standard normal if no modalities
            batch_size = 1
            mu = torch.zeros(batch_size, self.latent_dim, device=mu_list[0].device)
            logvar = torch.zeros(batch_size, self.latent_dim, device=mu_list[0].device)
            return mu, logvar
        
        batch_size = mu_list[0].size(0)
        device = mu_list[0].device
        
        # Convert logvar to variance
        var_list = [torch.exp(logvar) for logvar in logvar_list]
        
        # Compute precision-weighted sum
        precision_sum = torch.zeros(batch_size, self.latent_dim, device=device)
        weighted_mu_sum = torch.zeros(batch_size, self.latent_dim, device=device)
        
        for mu, var in zip(mu_list, var_list):
            precision = 1.0 / (var + 1e-8)
            precision_sum += precision
            weighted_mu_sum += precision * mu
        
        # PoE parameters
        mu_poe = weighted_mu_sum / (precision_sum + 1e-8)
        var_poe = 1.0 / (precision_sum + 1e-8)
        logvar_poe = torch.log(var_poe + 1e-8)
        
        return mu_poe, logvar_poe
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self,
                cgm: Optional[torch.Tensor] = None,
                insulin: Optional[torch.Tensor] = None,
                physiology: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multimodal inputs.
        
        Args:
            cgm: (batch_size, 288) or None
            insulin: (batch_size, 288) or None
            physiology: (batch_size, physiology_dim) or None
        
        Returns:
            dict with:
                - cgm_recon: reconstructed CGM
                - insulin_recon: reconstructed insulin
                - physiology_recon: reconstructed physiology
                - mu: PoE mean
                - logvar: PoE log-variance
        """
        mu_list = []
        logvar_list = []
        
        # Encode available modalities
        if cgm is not None:
            mu_cgm, logvar_cgm = self.cgm_encoder(cgm)
            mu_list.append(mu_cgm)
            logvar_list.append(logvar_cgm)
        
        if insulin is not None:
            mu_insulin, logvar_insulin = self.insulin_encoder(insulin)
            mu_list.append(mu_insulin)
            logvar_list.append(logvar_insulin)
        
        if physiology is not None:
            x_phys = self.physiology_encoder(physiology)
            mu_phys = self.physiology_mu(x_phys)
            logvar_phys = self.physiology_logvar(x_phys)
            mu_list.append(mu_phys)
            logvar_list.append(logvar_phys)
        
        # Product-of-Experts fusion
        if len(mu_list) == 0:
            raise ValueError("At least one modality must be provided")
        
        mu, logvar = self.product_of_experts(mu_list, logvar_list)
        z = self.reparameterize(mu, logvar)
        
        # Decode all modalities
        result = {
            'mu': mu,
            'logvar': logvar,
            'z': z,
        }
        
        if cgm is not None:
            result['cgm_recon'] = self.cgm_decoder(z)
        
        if insulin is not None:
            result['insulin_recon'] = self.insulin_decoder(z)
        
        if physiology is not None:
            result['physiology_recon'] = self.physiology_decoder(z)
        
        return result
    
    def encode(self,
               cgm: Optional[torch.Tensor] = None,
               insulin: Optional[torch.Tensor] = None,
               physiology: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get latent representation from available modalities."""
        out = self.forward(cgm, insulin, physiology)
        return out['z']
    
    def decode_cgm(self, z: torch.Tensor) -> torch.Tensor:
        """Decode CGM from latent."""
        return self.cgm_decoder(z)
    
    def decode_insulin(self, z: torch.Tensor) -> torch.Tensor:
        """Decode insulin from latent."""
        return self.insulin_decoder(z)
    
    def decode_physiology(self, z: torch.Tensor) -> torch.Tensor:
        """Decode physiology from latent."""
        return self.physiology_decoder(z)
