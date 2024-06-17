import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("models/unet_model.pth"))
model.eval()

# Define transformation
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


# Function to visualize predictions
def visualize_prediction(image_path, model, transform, device):
    # Load and preprocess image
    image = Image.open(image_path).convert("L")
    input_image = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_image)
        output = torch.sigmoid(output).cpu().squeeze().numpy()

    # Plot the input image and the predicted mask
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.array(image), cmap="gray")
    ax[0].set_title("Input Image")
    ax[1].imshow(output, cmap="gray")
    ax[1].set_title("Predicted Mask")
    plt.show()


# Example usage
image_path = "data/images/ISIC_0027009.jpg"
visualize_prediction(image_path, model, transform, device)
