import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load("models/unet_model.pth"))
model.eval()

# Define transformation
transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# Load and preprocess image
image = Image.open("data/images/image1.jpg").convert("L")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    output = torch.sigmoid(output).cpu().squeeze().numpy()

# Display result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(image.cpu().squeeze(), cmap="gray")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask")
plt.imshow(output, cmap="gray")

plt.show()
