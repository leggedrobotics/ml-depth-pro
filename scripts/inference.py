import os
import numpy as np
from PIL import Image
import depth_pro
from tqdm import tqdm

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval().to(device)

# Function to normalize depth values between 0.1 and 100 meters
def normalize_depth(depth, min_depth=0.1, max_depth=100):
    clipped_depth = np.clip(depth, min_depth, max_depth)
    normalized_depth = (clipped_depth - min_depth) / (max_depth - min_depth)
    return normalized_depth

# Function to log normalize depth values between 0.1 and 100 meters
def log_normalize_depth(depth, min_depth=0.1, max_depth=100):
    clipped_depth = np.clip(depth, min_depth, max_depth)
    log_depth = np.log(clipped_depth)
    log_min = np.log(min_depth)
    log_max = np.log(max_depth)
    normalized_log_depth = (log_depth - log_min) / (log_max - log_min)
    return normalized_log_depth

# Function to save image using PIL
def save_image(image_array, path):
    image = Image.fromarray(image_array)
    image.save(path)

# Process a folder of images and save depth predictions
def process_folder(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG"):
            image_path = os.path.join(input_dir, filename)
            output_base = os.path.join(output_dir, os.path.splitext(filename)[0])

            # Load and preprocess an image
            image, _, f_px = depth_pro.load_rgb(image_path)
            image = transform(image).to(device)

            # Run inference
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"].cpu().numpy()  # Depth in [m]

            # Save raw depth as .npz
            np.savez_compressed(f"{output_base}_depth.npz", depth.astype(np.float32))

            # min_max_norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
            # min_max_norm_depth_image = (min_max_norm_depth * 255).astype(np.uint8)
            # save_image(min_max_norm_depth_image, f"{output_base}_minmax.png")

            # # Normalize depth between 0.1 and 100 meters
            # normalized_depth = normalize_depth(depth)

            # # Log-normalize depth between 0.1 and 100 meters
            # log_normalized_depth = log_normalize_depth(depth)

            # # Save the normalized depth image (normalization)
            # depth_normalized_image = (normalized_depth * 255).astype(np.uint8)
            # save_image(depth_normalized_image, f"{output_base}_normalized.png")

            # # Save the log-normalized depth image
            # depth_log_normalized_image = (log_normalized_depth * 255).astype(np.uint8)
            # save_image(depth_log_normalized_image, f"{output_base}_log_normalized.png")

# Example usage
input_folder = "/media/patelm/ssd/imagenet-1k/test"
output_folder = "/media/patelm/ssd/imagenet-1k-ml-depth-pro/test"
process_folder(input_folder, output_folder)
