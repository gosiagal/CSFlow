"""GPU-accelerated data loading for ImageNet."""

import os
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path


class DataLoaderImagenet:
    """
    Efficient ImageNet dataloader with GPU transfer and batching.
    """
    
    def __init__(self, data_dir, resolution=256, batch_size=32, max_images=0):
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.max_images = max_images
        
        # Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Scale to [-1, 1]
        ])
        
        # Collect image paths
        self.image_paths = self._collect_image_paths()
        self.num_images = len(self.image_paths)
        
        print(f"Found {self.num_images} images in {data_dir}")
    
    def _collect_image_paths(self):
        """Collect all image file paths from ImageNet directory structure."""
        image_paths = []
        
        # ImageNet train structure: data_dir/class_folder/*.JPEG
        data_path = Path(self.data_dir)
        
        for class_folder in sorted(data_path.iterdir()):
            if not class_folder.is_dir():
                continue
            
            for img_path in sorted(class_folder.glob("*.JPEG")):
                image_paths.append(str(img_path))
                
                if self.max_images > 0 and len(image_paths) >= self.max_images:
                    return image_paths
        
        return image_paths
    
    def __iter__(self):
        """Iterate over batches of images on GPU."""
        batch = []
        
        for img_path in self.image_paths:
            try:
                # Load and preprocess image
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img)
                batch.append(img_tensor)
                
                # Yield batch when full
                if len(batch) == self.batch_size:
                    batch_tensor = torch.stack(batch).to(self.device)
                    yield batch_tensor
                    batch = []
            
            except Exception as e:
                print(f"Warning: Failed to load {img_path}: {e}")
                continue
        
        # Yield remaining images
        if len(batch) > 0:
            batch_tensor = torch.stack(batch).to(self.device)
            yield batch_tensor
    
    def __len__(self):
        """Number of batches."""
        return (self.num_images + self.batch_size - 1) // self.batch_size
