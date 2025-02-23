import os
import numpy as np
from PIL import Image

from tqdm import tqdm
import glob

import depth_pro
import torch
import argparse

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

    filenames = glob.glob(os.path.join(input_dir, '**/*'), recursive=True)

    os.makedirs(output_dir, exist_ok=True)

    filenames = [f for f in filenames if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    # filenames = [f for f in filenames if int(f.split('/')[-1].split('.')[0]) > 0 and int(f.split('/')[-1].split('.')[0]) < 31]

    print(filenames)

    filenames = sorted(filenames)

    
    for filename in tqdm(filenames, desc="Processing images"):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPEG"):
            image_path = os.path.join(input_dir, filename)
            pre = filename.split('/')[-2]
            os.makedirs(os.path.join(output_dir, pre), exist_ok=True)

            output_base = os.path.join(output_dir, pre, (filename.split('/')[-1]).split('.')[0])

            # Load and preprocess an image
            image, _, f_px = depth_pro.load_rgb(image_path)
            image = transform(image).to(device)

            # Run inference
            prediction = model.infer(image, f_px=f_px)
            depth = prediction["depth"].cpu().numpy()  # Depth in [m]

            # Save raw depth as .npz
            np.savez_compressed(f"{output_base}_depth.npz", depth.astype(np.float32))
            
            # cutoff depth at 100m
            depth[depth > 60] = 60
            min_max_norm_depth = (depth - depth.min()) / (depth.max() - depth.min())
            min_max_norm_depth_image = (min_max_norm_depth * 255).astype(np.uint8)

            save_image(min_max_norm_depth_image, f"{output_base}_depth.png")

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


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process images and save depth predictions.")
    parser.add_argument('--srcdir', type=str, required=True, help="Input directory containing images.")
    parser.add_argument('--outdir', type=str, required=True, help="Output directory for depth predictions.")
    return parser.parse_args()

def main():
    args = parse_args()

    input_folder = args.srcdir
    output_folder = args.outdir

    print(f"🔹 Input Directory: {input_folder}")
    print(f"🔹 Output Directory: {output_folder}")

    # Start processing
    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()