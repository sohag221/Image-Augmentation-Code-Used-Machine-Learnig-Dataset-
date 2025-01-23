import os
import cv2
from albumentations import (
    Compose, HorizontalFlip, RandomBrightnessContrast, ShiftScaleRotate
)

# Define the augmentation pipeline
augment = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
])

def augment_images(input_dir, output_dir, target_count):
    os.makedirs(output_dir, exist_ok=True)
    images = os.listdir(input_dir)
    current_count = len(images)
    augment_count = target_count - current_count

    print(f"Starting augmentation to reach {target_count} images...")
    count = 0

    while current_count < target_count:
        for img_file in images:
            if current_count >= target_count:
                break

            img_path = os.path.join(input_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Error reading image: {img_path}")
                continue

            # Apply augmentation
            augmented = augment(image=image)['image']
            output_path = os.path.join(output_dir, f'aug_{count}_{img_file}')
            cv2.imwrite(output_path, augmented)

            current_count += 1
            count += 1

    print(f"Augmentation complete. Total images: {current_count}")

# Directories
input_dir = "D:\Soil Classification Project\Soil Classification\lal soil"  # Input directory for clay soil
output_dir = "D:\Soil Classification Project\Soil Classification\lal soil augmented"  # Output directory for augmented images

# Target count
target_count = 900

# Run augmentation
augment_images(input_dir, output_dir, target_count)
