import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======== SETTINGS ========
model_path = r"C:\Users\BHAVANA\Downloads\ALML\crack_detector.pth"
image_path = r"C:\Users\BHAVANA\OneDrive\Desktop\CIVIL  PROJECT\organized_dataset\val\cracked\00043.jpg" # Replace with your image path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== LOAD MODEL ========
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ======== DEFINE TRANSFORM ========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======== CLASS MAPPING (from training) ========
# Adjust this based on train_dataset.class_to_idx output
# Example: {'crack': 0, 'no_crack': 1} or {'no_crack': 0, 'crack': 1}
class_map = {0: "crack", 1: "no_crack"}  # <- Change if needed

# ======== LOAD AND TRANSFORM IMAGE ========
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0).to(device)

# ======== PREDICT ========
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    pred_class = class_map[predicted.item()]
    print(f"✅ Predicted Class: {pred_class.capitalize()}")

# ======== VISUALIZATION WITH OPENCV ========
# Load with OpenCV for crack highlighting
cv_img = cv2.imread(image_path)
cv_img = cv2.resize(cv_img, (224, 224))

# Convert to grayscale and apply edge detection
gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

# Convert edges to color for visualization
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Blend with original image
overlay = cv2.addWeighted(cv_img, 0.7, edges_colored, 0.3, 0)

# Display using matplotlib
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Detected: {pred_class.capitalize()}")
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
