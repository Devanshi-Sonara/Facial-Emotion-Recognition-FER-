# utils/data_augmentation.py

from torchvision import transforms
from config import *

def get_augmenter():
    """
    Create a transform for data augmentation.
    """
    
    return transforms.Compose([
        # transforms.RandomRotation(ROTATION_RANGE),
        # transforms.RandomHorizontalFlip(HORIZONTAL_FLIP),
        # transforms.RandomAffine(0, translate=(WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(size=(48, 48)),  # Random cropping
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    
    ])