import laspy
import numpy as np
import h5py
from tqdm import tqdm
import os

def create_dataset(train_files, test_files, val_files, tile_size, overlap, grid_size=0.1, output_file='output.h5'):
    """
    Creates a dataset from .las files suitable for PointTransformerV3 model and saves it in HDF5 format.

    Args:
        train_files (list): List of paths to .las files for training.
        test_files (list): List of paths to .las files for testing.
        val_files (list): List of paths to .las files for validation.
        tile_size (float): Size of each tile in the same units as the point coordinates.
        overlap (float): Overlap between tiles.
        grid_size (float): Grid size for voxelization (default: 0.1).
        output_file (str): Path to the output HDF5 file.

    Returns:
        None
    """

    # Open or create the HDF5 file
    with h5py.File(output_file, 'w') as hdf:
        # Process each dataset split
        for split_name, las_files in zip(['train', 'test', 'val'], [train_files, test_files, val_files]):
            if not las_files:
                continue  # Skip if the list is empty

            print(f"Processing {split_name} files...")
            split_group = hdf.create_group(split_name)

            for las_file in tqdm(las_files, desc=f"Processing {split_name.upper()} LAS files"):
                las_filename = os.path.basename(las_file)
                las_group = split_group.create_group(las_filename)

                # Read the .las file
                with laspy.open(las_file) as f:
                    las = f.read()
                    coords = np.vstack((las.x, las.y, las.z)).transpose()  # Shape: [N, 3]

                    # Assuming the features are RGB colors
                    if all(dim in las.point_format.dimension_names for dim in ['red', 'green', 'blue']):
                        feats = np.vstack((las.red, las.green, las.blue)).transpose()  # Shape: [N, 3]
                        feats = feats / 65535.0  # Normalize RGB values
                    else:
                        # If no color, use intensity or create dummy features
                        if 'intensity' in las.point_format.dimension_names:
                            feats = las.intensity.reshape(-1, 1)  # Shape: [N, 1]
                            feats = feats / np.max(feats)  # Normalize intensity
                        else:
                            feats = np.ones((coords.shape[0], 1))  # Dummy feature

                    # Assuming 'classification' labels are available; if not, create dummy labels
                    if 'classification' in las.point_format.dimension_names:
                        target = las.classification  # Shape: [N]
                    else:
                        target = np.zeros(coords.shape[0], dtype=np.int32)  # Dummy labels

                # Get the bounding box of the point cloud
                min_coords = np.min(coords, axis=0)
                max_coords = np.max(coords, axis=0)

                # Compute the tile indices
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords

                x_range = np.arange(x_min, x_max, tile_size - overlap)
                y_range = np.arange(y_min, y_max, tile_size - overlap)

                tile_count = 0  # To name tile groups sequentially

                for x_start in x_range:
                    for y_start in y_range:
                        x_end = x_start + tile_size
                        y_end = y_start + tile_size

                        # Select points within the current tile
                        mask = (
                            (coords[:, 0] >= x_start) & (coords[:, 0] < x_end) &
                            (coords[:, 1] >= y_start) & (coords[:, 1] < y_end)
                        )
                        tile_coords = coords[mask]
                        tile_feats = feats[mask]
                        tile_target = target[mask]

                        if tile_coords.shape[0] == 0:
                            continue  # Skip empty tiles

                        # Create grid coordinates for voxelization (if needed)
                        grid_coord = np.floor(tile_coords / grid_size).astype(int)

                        # Index in the original cloud
                        idx_in_original_cloud = np.nonzero(mask)[0].astype(np.int32)

                        # Create a group for each tile
                        tile_name = f"{tile_count:05d}"
                        tile_group = las_group.create_group(tile_name)

                        # Save datasets in the tile group
                        tile_group.create_dataset('idx_in_original_cloud', data=idx_in_original_cloud, dtype='int32')
                        tile_group.create_dataset('pos', data=tile_coords, dtype='float32')
                        tile_group.create_dataset('feat', data=tile_feats, dtype='float32')
                        tile_group.create_dataset('target', data=tile_target, dtype='int32')

                        tile_count += 1  # Increment tile count

        print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    # Example usage:

    # Define your lists of LAS files for each dataset split
    train_files = [
        '/home/aerotract/NAS/main/Clients/10078_Newnastest2/CPI_LiDAR_Training_Data3/wyo1.las',
        '/home/aerotract/NAS/main/Clients/10078_Newnastest2/CPI_LiDAR_Training_Data3/wyo2.las',
        # Add more training files as needed
    ]

    test_files = [
        '/home/aerotract/NAS/main/Clients/10078_Newnastest2/CPI_LiDAR_Training_Data3/wyo3.las',
        # Add more testing files as needed
    ]

    val_files = [
        '/home/aerotract/NAS/main/Clients/10078_Newnastest2/CPI_LiDAR_Training_Data3/wyo4.las',
        # Add more validation files as needed
    ]

    tile_size = 10.0  # Example tile size
    overlap = 2.0     # Example overlap

    create_dataset(train_files, test_files, val_files, tile_size, overlap)

