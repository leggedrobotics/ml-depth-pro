import os
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DIR = "/home/patelm/docker_ws/SurroundOcc/figure_data_neighborhood/depth"
OUTPUT_DIR = "/home/patelm/docker_ws/SurroundOcc/figure_data_neighborhood/pcd_depth"
DEPTH_FOLDERS = {
    "CAM_FRONT": "FRONT_CAM",
    "CAM_FRONT_LEFT": "FRONT_LEFT_CAM",
    "CAM_FRONT_RIGHT": "FRONT_RIGHT_CAM",
    "CAM_BACK": "BACK_CAM",
    "CAM_BACK_LEFT": "BACK_LEFT_CAM",
    "CAM_BACK_RIGHT": "BACK_RIGHT_CAM",
}
MAX_DEPTH = 50.0  # Maximum depth threshold (meters)

# Function to calculate focal length from FOV
def calculate_focal_length(fov_x, width, height):
    fov_x_rad = np.deg2rad(fov_x)
    fx = (width / 2) / np.tan(fov_x_rad / 2)
    fov_y_rad = 2 * np.arctan((height / width) * np.tan(fov_x_rad / 2))
    fy = (height / 2) / np.tan(fov_y_rad / 2)
    return fx, fy

# Image resolution
width = 1600
height = 900

# FOVs
fov_front = 90  # Front, Front-Left, Front-Right, Back-Left, Back-Right
fov_back = 110  # Back camera


# Focal lengths
fx_front, fy_front = calculate_focal_length(fov_front, width, height)
fx_back, fy_back = calculate_focal_length(fov_back, width, height)

# # === CAMERA INTRINSICS ===
# width, height = 1280, 720
# fx_front, fy_front = 600, 600  # Placeholder focal lengths for front cameras
# fx_back, fy_back = 700, 700  # Placeholder focal lengths for back cameras

CAM_INTRINSICS = {
    "CAM_FRONT": np.array([[fx_front, 0, width // 2], [0, fy_front, height // 2], [0, 0, 1]]),
    "CAM_FRONT_LEFT": np.array([[fx_front, 0, width // 2], [0, fy_front, height // 2], [0, 0, 1]]),
    "CAM_FRONT_RIGHT": np.array([[fx_front, 0, width // 2], [0, fy_front, height // 2], [0, 0, 1]]),
    "CAM_BACK": np.array([[fx_back, 0, width // 2], [0, fy_back, height // 2], [0, 0, 1]]),
    "CAM_BACK_LEFT": np.array([[fx_front, 0, width // 2], [0, fy_front, height // 2], [0, 0, 1]]),
    "CAM_BACK_RIGHT": np.array([[fx_front, 0, width // 2], [0, fy_front, height // 2], [0, 0, 1]]),
}

# === CAMERA TO LIDAR ROTATIONS ===
CAM_ROTATIONS = {
    "CAM_FRONT": R.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix(),
    "CAM_FRONT_LEFT": R.from_euler("xyz", [90, 55, 0], degrees=True).as_matrix(),
    "CAM_FRONT_RIGHT": R.from_euler("xyz", [90, -55, 0], degrees=True).as_matrix(),
    "CAM_BACK": R.from_euler("xyz", [90, 180, 0], degrees=True).as_matrix(),
    "CAM_BACK_LEFT": R.from_euler("xyz", [90, 110, 0], degrees=True).as_matrix(),
    "CAM_BACK_RIGHT": R.from_euler("xyz", [90, -110, 0], degrees=True).as_matrix(),
}

# === FUNCTIONS ===
def load_depth_image(image_path):
    """
    Load depth image as a float32 numpy array.
    Supports both PNG and NPY formats.
    """
    if image_path.endswith(".npy"):
        return np.load(image_path)
    else:
        depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        return depth.astype(np.float32) / 1000.0  # Convert millimeters to meters

def depth_to_point_cloud(depth, K, max_depth=50.0, edge_threshold=0.3):
    """
    Convert depth image into a 3D point cloud, filtering sharp discontinuities to reduce bleeding.
    
    Args:
        depth: (H, W) Depth image.
        K: (3x3) Intrinsic matrix.
        max_depth: Maximum depth to consider.
        edge_threshold: Maximum allowed depth difference for valid points.
    
    Returns:
        points: (N, 3) 3D coordinates.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Compute depth gradients (edges)
    depth_dx = np.abs(np.gradient(depth, axis=1))
    depth_dy = np.abs(np.gradient(depth, axis=0))
    depth_edges = (depth_dx + depth_dy) > edge_threshold  # Mask large depth changes

    # Create valid mask: remove points near depth discontinuities
    valid_mask = (depth > 0) & (depth < max_depth) & ~depth_edges
    depth_filtered = depth[valid_mask]
    u = u[valid_mask]
    v = v[valid_mask]
    
    x = (u - K[0, 2]) * depth_filtered / K[0, 0]
    y = (v - K[1, 2]) * depth_filtered / K[1, 1]
    z = depth_filtered

    return np.stack([x, y, z], axis=1)

def transform_to_lidar(points, cam_name):
    """
    Transform 3D points from camera frame to Lidar frame.
    
    Args:
        points: (N, 3) 3D points in the camera frame.
        cam_name: Name of the camera (e.g., "CAM_FRONT").
    
    Returns:
        Transformed (N, 3) points in the Lidar frame.
    """
    # Create 4x4 transformation matrix
    cam_to_lidar = np.eye(4)
    cam_to_lidar[:3, :3] = np.linalg.inv(CAM_ROTATIONS[cam_name])  # Apply rotation

    # Convert points to homogeneous coordinates
    points_homo = np.hstack((points, np.ones((points.shape[0], 1))))  # (N, 4)

    # Apply transformation
    points_lidar = (cam_to_lidar @ points_homo.T)[:3].T  # Convert back to (N, 3)
    
    return points_lidar

# === MAIN FUNCTION ===
def main():
    # Get all sample indices based on the first camera folder
    first_cam_folder = os.path.join(ROOT_DIR, list(DEPTH_FOLDERS.values())[0])
    sample_indices = sorted([f.split("_")[0] for f in os.listdir(first_cam_folder) if f.endswith(".npy")])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Iterate through each sample index
    for sample_idx in tqdm(sample_indices, desc="🔹 Processing Samples"):
        merged_pcd = o3d.geometry.PointCloud()

        # Iterate through all cameras for this sample
        for cam_name, folder in DEPTH_FOLDERS.items():
            depth_path = os.path.join(ROOT_DIR, folder, f"{sample_idx}_raw_depth_meter.npy")
            if not os.path.exists(depth_path):
                continue  # Skip if file is missing

            # Load depth image
            depth = load_depth_image(depth_path)

            # Convert depth to 3D points (Camera frame)
            points_cam = depth_to_point_cloud(depth, CAM_INTRINSICS[cam_name], MAX_DEPTH)

            # Transform to Lidar frame
            points_lidar = transform_to_lidar(points_cam, cam_name)

            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_lidar)

            # Merge into sample-specific point cloud
            merged_pcd += pcd

        # Downsample the merged point cloud
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.5)

        # Save point cloud for this sample
        output_pcd_path = os.path.join(OUTPUT_DIR, f"{sample_idx}.pcd")
        o3d.io.write_point_cloud(output_pcd_path, merged_pcd)
        print(f"✅ Saved: {output_pcd_path}")

    print("✅ All samples processed.")

if __name__ == "__main__":
    main()