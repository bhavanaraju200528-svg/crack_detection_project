import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import time

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transforms
print("Defining image transformations...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 input size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
data_dir = r"C:\Users\BHAVANA\Downloads\ALML\organized_dataset"  # Change this to your dataset path
print("Loading datasets...")
train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
print(f"Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

# Load Pretrained ResNet50
print("Loading pre-trained ResNet50 model...")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: cracked, non-cracked
model = model.to(device)
print("Model modified for binary classification.")

# Define loss and optimizer
print("Defining loss function and optimizer...")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Change as needed
train_acc, val_acc = [], []
print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs} ({(epoch + 1) / num_epochs * 100:.2f}% completed):")
    model.train()
    running_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress = (batch_idx + 1) / len(train_loader) * 100
        print(f"  Epoch {epoch + 1}/{num_epochs} - {progress:.2f}% completed - Loss: {loss.item():.4f}")

    train_acc.append(correct / total)
    print(f"  Training accuracy: {train_acc[-1] * 100:.2f}%")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_acc.append(correct / total)
    print(f"  Validation accuracy: {val_acc[-1] * 100:.2f}%")

# Save Model
print("Saving trained model...")
torch.save(model.state_dict(), "crack_detector.pth")
print("Model saved as 'crack_detector.pth'.")

# Plot accuracy
print("Plotting training and validation accuracy...")
plt.plot(range(1, num_epochs + 1), train_acc, label="Train Accuracy")
plt.plot(range(1, num_epochs + 1), val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
print("Training process completed!")
