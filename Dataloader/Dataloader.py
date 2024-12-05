import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader

class H5Dataset(Dataset):
    def __init__(self, h5_file_path, split='train'):
        self.h5_file_path = h5_file_path
        self.split = split

        # Open the HDF5 file
        self.h5_file = h5py.File(h5_file_path, 'r')

        # Collect all tile datasets under the specified split
        self.tiles = []
        split_group = self.h5_file[split]
        for las_file in split_group:
            las_group = split_group[las_file]
            for tile_name in las_group:
                tile_path = f"{split}/{las_file}/{tile_name}"
                self.tiles.append(tile_path)

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        tile_path = self.tiles[idx]
        tile_group = self.h5_file[tile_path]

        # Load data
        pos = tile_group['pos'][:]          # Shape: [N, 3]
        feat = tile_group['feat'][:]        # Shape: [N, C]
        target = tile_group['target'][:]    # Shape: [N]

        # Convert to tensors
        pos = torch.tensor(pos, dtype=torch.float32)
        feat = torch.tensor(feat, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.long)

        # Create a data dictionary
        data_dict = {
            'pos': pos,        # Positions
            'feat': feat,      # Features
            'target': target,  # Labels
        }

        return data_dict

def collate_fn(batch):
    # Batch is a list of data_dicts
    batch_pos = []
    batch_feat = []
    batch_target = []
    batch_offset = []

    cumsum = 0
    for data in batch:
        N = data['pos'].shape[0]
        batch_pos.append(data['pos'])
        batch_feat.append(data['feat'])
        batch_target.append(data['target'])
        cumsum += N
        batch_offset.append(cumsum)

    batch_pos = torch.cat(batch_pos, dim=0)         # Shape: [Total_N, 3]
    batch_feat = torch.cat(batch_feat, dim=0)       # Shape: [Total_N, C]
    batch_target = torch.cat(batch_target, dim=0)   # Shape: [Total_N]
    batch_offset = torch.tensor(batch_offset)       # Shape: [Batch_size]

    batch_data = {
        'pos': batch_pos,
        'feat': batch_feat,
        'target': batch_target,
        'offset': batch_offset
    }

    return batch_data

def get_dataloader(h5_file_path, split='train', batch_size=4):
    dataset = H5Dataset(h5_file_path, split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    return dataloader

if __name__ == "__main__":
    h5_file_path = 'output.h5'  # Path to your HDF5 file
    batch_size = 4

    # Get the DataLoader
    dataloader = get_dataloader(h5_file_path, split='train', batch_size=batch_size)

    # Example loop over the DataLoader
    for batch_data in dataloader:
        # Access the data
        pos = batch_data['pos']          # [Total_N, 3]
        feat = batch_data['feat']        # [Total_N, C]
        target = batch_data['target']    # [Total_N]
        offset = batch_data['offset']    # [Batch_size]

        data_dict = {
            'pos': pos,
            'feat': feat,
            'target': target,
            'offset': offset
        }

        # Assuming your model is defined as 'model'
        # and it accepts 'pos', 'feat', and 'offset' as inputs
        # Output might be per point or per batch
        # output = model(pos, feat, offset)

        # Compute loss, backpropagation, etc.
        # loss = criterion(output, target)
        # loss.backward()
        # optimizer.step()

        # For demonstration, we'll just print the shapes
        print(f"pos shape: {pos.shape}")
        print(f"feat shape: {feat.shape}")
        print(f"target shape: {target.shape}")
        print(f"offset: {offset}")
        print()
        # break  # Remove this break to process the entire dataset
