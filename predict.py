import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("models/unet_model.pth"))
model.eval()

# Define transformation
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


# Function to visualize predictions and apply edge detection
def visualize_prediction(image_path, model, transform, device):
    # Load and preprocess image
    image = Image.open(image_path).convert("L")
    input_image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_image)
        output = torch.sigmoid(output).cpu().squeeze().numpy()

    # Preprocess the predicted mask
    mask = (output * 255).astype(np.uint8)
    blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Apply edge detection with adjusted parameters
    edges = cv2.Canny(blurred_mask, 50, 150)

    # Apply morphological operations to refine edges
    kernel = np.ones((3, 3), np.uint8)
    refined_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Plot the input image, predicted mask, and edge-detected mask
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(np.array(image), cmap="gray")
    ax[0].set_title("Input Image")
    ax[1].imshow(output, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[2].imshow(refined_edges, cmap="gray")
    ax[2].set_title("Edge-Detected Mask")
    plt.show()


# Example usage
image_path = "data/images/ISIC_0027009.jpg"  # Use an actual filename from your dataset
visualize_prediction(image_path, model, transform, device)
