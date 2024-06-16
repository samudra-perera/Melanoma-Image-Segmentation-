import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet
from utils import SegmentationDataset, train_test_split_dataset


def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Define transformations
        transform = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor()]
        )

        # Load dataset
        print("Loading dataset...")
        dataset = SegmentationDataset(
            image_dir="data/images", mask_dir="data/masks", transform=transform
        )
        train_dataset, test_dataset = train_test_split_dataset(dataset, test_size=0.2)
        print(
            f"Dataset loaded. Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}"
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Initialize model, loss, and optimizer
        print("Initializing model...")
        model = UNet(in_channels=1, out_channels=1).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Number of epochs
        num_epochs = 50

        # Training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            epoch_loss = 0
            for images, masks in train_loader:
                images = images.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()  # Clear previous gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, masks)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update parameters

                epoch_loss += loss.item()  # Accumulate loss

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}"
            )

        # Save the trained model
        print("Saving the trained model...")
        torch.save(model.state_dict(), "models/unet_model.pth")

        # Evaluate the model on the test set
        print("Evaluating the model on the test set...")
        model.eval()  # Set model to evaluation mode
        test_loss = 0
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
