import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Define image size
IMAGE_SIZE = (299, 299)

# Define a basic transform (normal processing for validation)
def basic_transform(image):
    # Convert the input PIL image to numpy array
    image_np = np.array(image)
    # Resize
    image_np = cv2.resize(image_np, IMAGE_SIZE)
    # Normalize using ImageNet statistics
    image_np = image_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    # Convert to uint8 for display purposes (optional)
    # Here we convert it back to a displayable image by re-scaling to 0-1 (for visualization).
    return image_np

# Define the augmentation pipeline as used in your project
def apply_augmentation(image, image_size=IMAGE_SIZE):
    # Convert PIL Image to numpy array
    image_np = np.array(image)
    # If grayscale or with alpha channel, convert to 3-channel RGB
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    aug = A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent=0.0625, scale=(0.9, 1.1), rotate=(-45, 45), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.MedianBlur(blur_limit=5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1.0, p=0.5),
        ], p=0.3),
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    augmented = aug(image=image_np)
    # To visualize, convert the tensor back to numpy array (removing normalization)
    # Here we assume the tensor is normalized; for visualization, we reverse normalization:
    aug_img = augmented['image'].cpu().numpy().transpose(1, 2, 0)
    # Reverse normalization (approximate)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    aug_img = aug_img * std + mean
    aug_img = np.clip(aug_img, 0, 1)
    return aug_img

def visualize_normal_vs_augmented(image_path):
    # Load image
    image_pil = Image.open(Path(image_path)).convert('RGB')
    
    # Apply the basic transform for a "normal" processed image
    normal_img = basic_transform(image_pil)
    
    # Apply augmentation for the malignant version
    augmented_img = apply_augmentation(image_pil)
    
    # Plot side by side for comparison
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(normal_img)
    plt.title("Normal Processed Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_img)
    plt.title("Augmented Image (Malignant)")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("normal_vs_augmented.png")
    plt.show()

if __name__ == "__main__":
    # Altere este caminho para uma imagem de exemplo apropriada, por exemplo, de uma lesão maligna
    image_path = "data/processed/train/malignant/ISIC_0034317.jpg"
    visualize_normal_vs_augmented(image_path)