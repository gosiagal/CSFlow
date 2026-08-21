"""GPU-accelerated RAPSD calculator using batched PyTorch FFT."""

import torch
import numpy as np
from tqdm import tqdm

# Based on pysteps.utils.spectral.rapsd implementation, but optimized for GPU and batched processing.
# https://github.com/pySTEPS/pysteps/blob/master/pysteps/utils/spectral.py

class RapsdCalculatorGPU:
    """Radially Averaged Power Spectral Density calculator using batched GPU FFT."""
    
    def __init__(self, device, image_size=256):
        self.device = device
        self.image_size = image_size
        self.precomputed_bins = False
        self.bin_indices_flat = None
        self.bin_counts = None
        self.frequencies = None
        self.num_bins = None

    def _precompute_bins(self, h, w, d=1.0):
        """Precompute frequency bins and indices for radial averaging."""
        if self.precomputed_bins:
            return

        # Create frequency grid
        freq_y = torch.fft.fftfreq(h, d=d, device=self.device)
        freq_x = torch.fft.fftfreq(w, d=d, device=self.device)
        freq_grid_y, freq_grid_x = torch.meshgrid(freq_y, freq_x, indexing='ij')
        
        # Shift to match fftshift
        freq_grid_y = torch.fft.fftshift(freq_grid_y)
        freq_grid_x = torch.fft.fftshift(freq_grid_x)
        
        # Radial frequency magnitude
        freq_magnitude = torch.sqrt(freq_grid_x**2 + freq_grid_y**2)
        
        # Create radial bins
        max_freq = torch.max(freq_magnitude)
        num_bins = max(h, w) // 2
        bin_edges = torch.linspace(0, max_freq, num_bins + 1, device=self.device)
        
        # Digitize frequencies into bins
        # bucketize returns indices i where bin_edges[i-1] <= val < bin_edges[i]
        # We want 0-based bin indices, so we subtract 1.
        bin_indices = torch.bucketize(freq_magnitude, bin_edges, right=False) - 1
        
        # Clip to valid range [0, num_bins-1] to handle edge cases (e.g. max_freq)
        bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
        
        # Flatten for scatter_add
        self.bin_indices_flat = bin_indices.view(-1)
        
        # Count pixels per bin for averaging
        self.bin_counts = torch.bincount(self.bin_indices_flat, minlength=num_bins).float()
        
        # Avoid division by zero (though bins should generally be populated)
        self.bin_counts[self.bin_counts == 0] = 1.0
        
        # Frequencies (bin centers)
        self.frequencies = (bin_edges[:-1] + bin_edges[1:]) / 2
        self.num_bins = num_bins
        self.precomputed_bins = True

    def compute_rapsd_batch(self, images):
        """
        Compute RAPSD for a batch of images on GPU.
        
        Args:
            images: tensor of shape (B, H, W) or (B, C, H, W)
        
        Returns:
            rapsd_batch: tensor of shape (B, num_bins)
            frequencies: tensor of shape (num_bins,)
        """
        # Handle channel dimension - convert to grayscale
        if images.ndim == 4:
            # Rec. 601 weights: 0.299 R + 0.587 G + 0.114 B
            # Assumes images are (B, C, H, W)
            images = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            
        B, H, W = images.shape
        
        # Precompute bins if needed (assumes constant H, W)
        self._precompute_bins(H, W)
        
        # Compute 2D FFT
        fft_result = torch.fft.fft2(images)
        fft_shifted = torch.fft.fftshift(fft_result, dim=(-2, -1))
        
        # Power spectrum (normalized)
        power_spectrum = (torch.abs(fft_shifted) ** 2) / (H * W)
        
        # Flatten spatial dims: (B, H*W)
        power_spectrum_flat = power_spectrum.view(B, -1)
        
        # Sum power in each bin
        # We need to sum power_spectrum_flat into bins defined by self.bin_indices_flat
        # Expand bin indices to match batch size: (B, H*W)
        indices_expanded = self.bin_indices_flat.unsqueeze(0).expand(B, -1)
        
        rapsd_sums = torch.zeros(B, self.num_bins, device=self.device)
        rapsd_sums.scatter_add_(1, indices_expanded, power_spectrum_flat)
        
        # Average
        rapsd_means = rapsd_sums / self.bin_counts.unsqueeze(0)
        
        return rapsd_means, self.frequencies

    def compute_dataset_mean_rapsd(self, dataloader):
        """
        Compute mean RAPSD across entire dataset using GPU.
        """
        sum_rapsd = None
        count = 0
        frequencies = None
        
        # Ensure dataloader yields batches on GPU
        for batch in tqdm(dataloader, desc="Computing RAPSD (GPU)"):
            # batch shape: (B, C, H, W)
            
            rapsd_batch, freqs = self.compute_rapsd_batch(batch)
            
            if sum_rapsd is None:
                sum_rapsd = torch.zeros_like(rapsd_batch[0])
                frequencies = freqs
            
            sum_rapsd += rapsd_batch.sum(dim=0)
            count += batch.shape[0]
        
        mean_rapsd = sum_rapsd / count
        
        return mean_rapsd.cpu().numpy(), frequencies.cpu().numpy()
    
    def compute_noise_rapsd(self, num_samples, batch_size=64):
        """
        Generate noise and compute its mean RAPSD on GPU.
        """
        sum_rapsd = None
        count = 0
        frequencies = None
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Computing noise RAPSD (GPU)"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            
            # Generate noise on GPU
            noise_batch = torch.randn(current_batch_size, self.image_size, self.image_size, device=self.device)
            
            rapsd_batch, freqs = self.compute_rapsd_batch(noise_batch)
            
            if sum_rapsd is None:
                sum_rapsd = torch.zeros_like(rapsd_batch[0])
                frequencies = freqs
            
            sum_rapsd += rapsd_batch.sum(dim=0)
            count += current_batch_size
        
        mean_rapsd_noise = sum_rapsd / count
        
        return mean_rapsd_noise.cpu().numpy()
