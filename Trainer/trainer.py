import sys
sys.path.append('/app/PointTransformerV3/')
sys.path.append('/app/AeroTract_LiDAR_Pipeline/')
from model import PointTransformerV3
from Dataloader.Dataloader import get_dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter


if __name__ == "__main__":
    h5_file_path = '/app/AeroTract_LiDAR_Pipeline/output.h5'  # Path to your HDF5 file
    batch_size = 8
    num_classes = 6  # Specify number of classes here
    num_epochs = 50  # Set the number of epochs

    # Get the DataLoader
    dataloader = get_dataloader(h5_file_path, split='train', batch_size=batch_size)
    model = PointTransformerV3(in_channels=3, num_classes=num_classes).cuda()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_points = 0
        all_targets = []

        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to GPU
            pos = batch_data['pos'].cuda()          # [Total_N, 3]
            feat = batch_data['feat'].cuda()        # [Total_N, C]
            # Map target values from [2,5,17,27,28] to [0,1,2,3,4]
            target_mapping = {2: 0, 5: 1, 17: 2, 27: 3, 28: 4,7:5}
            target = torch.tensor([target_mapping[x.item()] for x in batch_data['target']]).cuda()    # [Total_N]
            offset = batch_data['offset'].cuda()    # [Batch_size]
            
            # Store targets for majority class calculation
            all_targets.extend(target.cpu().numpy())

            data_dict = {
                'coord': pos,
                'grid_size': 0.1,
                'feat': feat,
                'target': target,
                'offset': offset
            }

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data_dict)

            # Get logits from the classifier
            logits = output['logits']  # Shape: [Total_N, num_classes]

            # Compute loss
            loss = criterion(logits, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            total_correct += (predicted == target).sum().item()
            total_points += target.size(0)

            # Print statistics every 10 batches
            if batch_idx % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0

        # Calculate and print epoch accuracy
        epoch_accuracy = 100 * total_correct / total_points
        
        # Calculate majority class accuracy
        majority_class = Counter(all_targets).most_common(1)[0][0]
        majority_correct = sum(1 for t in all_targets if t == majority_class)
        majority_accuracy = 100 * majority_correct / total_points
        
        print(f'Epoch [{epoch+1}/{num_epochs}] Model Accuracy: {epoch_accuracy:.2f}%')
        print(f'Epoch [{epoch+1}/{num_epochs}] Majority Class Accuracy: {majority_accuracy:.2f}%')

    print('Training complete.')
