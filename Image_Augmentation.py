import Augmentor
import os

# Path to your main dataset directory
main_directory = "D:/ML Dataset Project"  # Update this path to your dataset location

# Target number of images per class
target_count = 1000

# Iterate through each class folder
for class_folder in os.listdir(main_directory):
    class_path = os.path.join(main_directory, class_folder)

    if os.path.isdir(class_path):  # Ensure it's a folder
        print(f"\nProcessing folder: {class_folder}")

        # Count the existing PNG images in the folder
        existing_images = [
            f for f in os.listdir(class_path)
            if os.path.isfile(os.path.join(class_path, f)) and os.path.splitext(f)[1].lower() == ".png"
        ]
        current_count = len(existing_images)
        print(f"Current image count for {class_folder}: {current_count}")

        if current_count == 0:
            print(f"Skipping {class_folder} because it has no valid images.")
            continue

        if current_count >= target_count:
            print(f"Skipping {class_folder} as it already has {current_count} images.")
            continue  # Skip if the folder already has 1000 or more images

        # Calculate the number of images to generate
        images_to_generate = target_count - current_count
        print(f"Need to generate {images_to_generate} images for {class_folder}")

        try:
            # Create a pipeline for the current class folder
            pipeline = Augmentor.Pipeline(class_path)

            # Add augmentation operations
            pipeline.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)  # Rotate images
            pipeline.flip_left_right(probability=0.5)  # Horizontal flip
            pipeline.zoom_random(probability=0.5, percentage_area=0.8)  # Random zoom
            pipeline.random_contrast(probability=0.5, min_factor=0.7, max_factor=1.3)  # Adjust contrast
            pipeline.random_brightness(probability=0.5, min_factor=0.7, max_factor=1.3)  # Adjust brightness
            pipeline.shear(probability=0.5, max_shear_left=10, max_shear_right=10)  # Shearing
            pipeline.random_color(probability=0.5, min_factor=0.7, max_factor=1.3)  # Random color

            # Generate the required number of images
            pipeline.sample(images_to_generate)

            # Move augmented images back to the main folder
            output_folder = os.path.join(class_path, "output")
            for augmented_image in os.listdir(output_folder):
                os.rename(os.path.join(output_folder, augmented_image), os.path.join(class_path, augmented_image))

            os.rmdir(output_folder)  # Remove the empty output folder
            print(f"Augmentation completed for {class_folder}. Total images: {target_count}")
        except Exception as e:
            print(f"An error occurred while processing {class_folder}: {e}")

print("\nDataset augmentation completed for all classes!")
