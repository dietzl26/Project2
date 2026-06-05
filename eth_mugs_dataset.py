"""ETH Mugs Dataset."""

import os
from PIL import Image
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from utils import IMAGE_SIZE, load_mask


# This is only an example - you DO NOT have to use it exactly in this form!
class ETHMugsDataset(Dataset):
    """Torch dataset template shared as an example."""

    def __init__(self, root_dir, mode="train", augment=False, image_ids=None):
        """This dataset class loads the ETH Mugs dataset.

        It will return the resized image according to the scale and mask tensors
        in the original resolution.

        Args:
            root_dir (str): Path to the root directory of the dataset.
            mode (str): Mode of the dataset. It can be "train", "val" or "test"
            augment (bool): Whether to apply data augmentation (train mode only).
            image_ids (list): List of image IDs to use from the dataset.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.augment = augment

        # Get image and mask paths
        self.rgb_dir = os.path.join(self.root_dir, "rgb")
        self.mask_dir = os.path.join(self.root_dir, "masks")
        
        # Build image paths from image_ids or from available files
        self.image_paths = []
        if image_ids is not None:
            # Use provided image IDs (already strings like "0000")
            for img_id in image_ids:
                rgb_path = os.path.join(self.rgb_dir, f"{img_id}_rgb.jpg")
                if os.path.exists(rgb_path):
                    self.image_paths.append(img_id)
        else:
            # Discover all available images
            if os.path.exists(self.rgb_dir):
                for fname in sorted(os.listdir(self.rgb_dir)):
                    if fname.endswith("_rgb.jpg"):
                        # Extract image ID (e.g., "0000" from "0000_rgb.jpg")
                        img_id = fname.replace("_rgb.jpg", "")
                        self.image_paths.append(img_id)

        # Set image transforms - resize to IMAGE_SIZE
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=Image.BILINEAR),
            transforms.ToTensor(),
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

        print("[INFO] Dataset mode:", mode)
        print(
            "[INFO] Number of images in the ETHMugDataset: {}".format(len(self.image_paths))
        )

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Get an item from the dataset."""
        img_id = self.image_paths[idx]
        
        # Load image
        rgb_path = os.path.join(self.rgb_dir, f"{img_id}_rgb.jpg")
        image = Image.open(rgb_path).convert("RGB")
        image = self.transform(image)

        if self.mode != "test":
            # Load ground truth mask
            mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.png")
            mask = load_mask(mask_path)
            mask = Image.fromarray((mask * 255).astype('uint8'))
            mask = self.mask_transform(mask)
            return image, mask

        # For test mode, return image_id instead of mask
        return image, img_id
