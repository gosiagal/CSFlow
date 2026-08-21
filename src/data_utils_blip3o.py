"""Data loader for BLIP3o-60k dataset in webdataset tar format."""

import io
import tarfile
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


class DataLoaderBLIP3o:
    """
    Dataloader for BLIP3o-60k dataset that reads directly from tar files.
    Extracts only images, ignoring text captions.
    """

    def __init__(self, data_dir, resolution=256, batch_size=32, max_images=0):
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.max_images = max_images

        # Check for CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image preprocessing
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W)
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Scale to [-1, 1]
            ]
        )

        # Collect tar files
        self.tar_paths = self._collect_tar_paths()
        print(f"Found {len(self.tar_paths)} tar files in {data_dir}")

        # Count total images
        self.num_images = self._count_images()
        print(f"Total images: {self.num_images}")

    def _collect_tar_paths(self):
        """Collect all tar file paths from the dataset directory."""
        data_path = Path(self.data_dir)
        tar_paths = sorted(data_path.glob("*.tar"))
        return [str(p) for p in tar_paths]

    def _count_images(self):
        """Count total number of images across all tar files."""
        total = 0

        for tar_path in self.tar_paths:
            try:
                with tarfile.open(tar_path, "r") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and self._is_image_file(member.name):
                            total += 1
                            if self.max_images > 0 and total >= self.max_images:
                                return total
            except Exception as e:
                print(f"Warning: Failed to read {tar_path}: {e}")
                continue

        return total

    def _is_image_file(self, filename):
        """Check if file is an image based on extension."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        return Path(filename).suffix.lower() in image_extensions

    def _read_images_from_tar(self, tar_path):
        """Generator that yields images from a tar file."""
        try:
            with tarfile.open(tar_path, "r") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or not self._is_image_file(member.name):
                        continue

                    try:
                        # Extract file data
                        f = tar.extractfile(member)
                        if f is None:
                            continue

                        # Read image bytes
                        image_bytes = f.read()

                        # Load image from bytes
                        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                        # Apply transforms
                        img_tensor = self.transform(img)

                        yield img_tensor

                    except Exception as e:
                        print(
                            f"Warning: Failed to process {member.name} in {tar_path}: {e}"
                        )
                        continue

        except Exception as e:
            print(f"Warning: Failed to open tar file {tar_path}: {e}")

    def __iter__(self):
        """Iterate over batches of images on GPU."""
        batch = []
        images_processed = 0

        for tar_path in self.tar_paths:
            for img_tensor in self._read_images_from_tar(tar_path):
                batch.append(img_tensor)
                images_processed += 1

                # Yield batch when full
                if len(batch) == self.batch_size:
                    batch_tensor = torch.stack(batch).to(self.device)
                    yield batch_tensor
                    batch = []

                # Stop if max_images reached
                if self.max_images > 0 and images_processed >= self.max_images:
                    # Yield remaining images
                    if len(batch) > 0:
                        batch_tensor = torch.stack(batch).to(self.device)
                        yield batch_tensor
                    return

        # Yield remaining images
        if len(batch) > 0:
            batch_tensor = torch.stack(batch).to(self.device)
            yield batch_tensor

    def __len__(self):
        """Number of batches."""
        return (self.num_images + self.batch_size - 1) // self.batch_size
